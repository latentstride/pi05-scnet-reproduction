"""Exact Orbax restore, native JAX LoRA partition and policy sampling."""
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import flax.nnx as nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from openpi import transforms
from openpi.models import gemma
from openpi.models import model as openpi_model
from openpi.training import config as openpi_config

from .adapter import NativeAdapter
from .config import MANIFEST, OPENPI, PACE_MODEL, SETTINGS
from .data import digest

LORA_MODULES = ("q_einsum", "kv_einsum", "attn_vec_einsum", "mlp", "q_einsum_1",
                "kv_einsum_1", "attn_vec_einsum_1", "mlp_1")
ATTN_LEAVES = ("lora_a", "lora_b")
MLP_LEAVES = ("gating_einsum_lora_a", "gating_einsum_lora_b", "linear_lora_a", "linear_lora_b")


def model_configs():
    pace = openpi_config.get_config("pi05_robotwin_handover_block").model
    if (pace.pi05, pace.discrete_state_input, pace.action_dim, pace.action_horizon,
            pace.paligemma_variant, pace.action_expert_variant) != (True, True, 32, 50, "gemma_2b", "gemma_300m"):
        raise ValueError("Frozen PACE configuration differs")
    return pace, dataclasses.replace(pace, paligemma_variant=SETTINGS.paligemma_variant,
                                     action_expert_variant=SETTINGS.action_expert_variant)


def validate_install(root, *, full_hash=False):
    source = root / "source" / f"Openpi-PACE-{OPENPI}"
    if Path(openpi_model.__file__).resolve() != source / "src/openpi/models/model.py":
        raise RuntimeError(f"Loaded unexpected OpenPI source: {openpi_model.__file__}")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    tracked = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--",
                                       "src", "pace", "packages/openpi-client"], cwd=source, text=True).strip()
    if rev != OPENPI or tracked:
        raise RuntimeError(f"Unexpected source revision or tracked changes: {rev} {tracked}")
    downloads = root / "cache/pace-handover"
    data_home = root / "cache/openpi-pace"
    if Path(os.environ.get("OPENPI_DATA_HOME", "")).resolve() != data_home:
        raise RuntimeError("OPENPI_DATA_HOME must point to the pinned tokenizer cache")
    manifest = json.loads(MANIFEST.read_text())
    if manifest["revision"] != PACE_MODEL:
        raise ValueError("Unexpected checkpoint manifest")
    for record in manifest["files"]:
        name = record["path"]
        path = downloads / name
        if path.stat().st_size != record["bytes"] or (full_hash and digest(path) != record["sha256"]):
            raise ValueError(f"Checkpoint manifest mismatch: {name}")
        if name == "tokenizer.model":
            tokenizer = data_home / "big_vision/paligemma_tokenizer.model"
            if digest(tokenizer) != record["sha256"]:
                raise ValueError("Cached tokenizer mismatch")
    return downloads / "handover_block/params"


def audit_graph(config):
    model = nnx.eval_shape(config.create, jax.random.key(0))
    graph, state = nnx.split(model)
    flat = flatten_dict(state.to_pure_dict())
    expected = {
        ("PaliGemma", "llm", "layers", "attn", module, leaf)
        for module in LORA_MODULES if not module.startswith("mlp")
        for leaf in ATTN_LEAVES
    } | {
        ("PaliGemma", "llm", "layers", module, leaf)
        for module in LORA_MODULES if module.startswith("mlp")
        for leaf in MLP_LEAVES
    }
    actual = {path for path in flat if "lora" in str(path[-1])}
    if actual != expected:
        raise ValueError(f"Unexpected LoRA graph: missing={expected - actual}, extra={actual - expected}")
    predicate = lambda path, value: path in actual and issubclass(value.type, nnx.Param)
    split_lora = flatten_dict(state.filter(predicate).to_pure_dict())
    if set(split_lora) != actual:
        raise ValueError("NNX optimizer partition misses a LoRA parameter")
    report = {}
    for name, paths in (("trainable", actual), ("frozen", set(flat) - actual)):
        entries = [{"path": "/".join(p), "shape": list(flat[p].shape),
                    "init_dtype": str(flat[p].dtype), "runtime_dtype": "float32" if name == "trainable" else "bfloat16",
                    "elements": math.prod(flat[p].shape)} for p in sorted(paths)]
        report[name] = {"elements": sum(row["elements"] for row in entries), "leaves": entries}
    for module in ("gemma_2b_lora", "gemma_300m_lora"):
        report[module] = {k: {"rank": v.rank, "alpha": v.alpha}
                          for k, v in gemma.get_config(module).lora_configs.items()}
    return graph, state, flat, actual, predicate, report


def check_shapes(reference, restored):
    if set(reference) != set(restored):
        raise ValueError(f"Orbax paths differ: missing={reference.keys() - restored.keys()}, "
                         f"extra={restored.keys() - reference.keys()}")
    for path, spec in reference.items():
        arr = restored[path]
        if spec.shape != arr.shape or not jnp.issubdtype(arr.dtype, jnp.floating):
            raise ValueError(f"Incompatible Orbax leaf {path}: {arr.shape}, {arr.dtype}")


def checkpoint_layout(params_path):
    pace, lora = model_configs()
    graph, state, flat, train_paths, predicate, report = audit_graph(lora)
    frozen_graph, frozen_state = nnx.split(nnx.eval_shape(pace.create, jax.random.key(0)))
    frozen_spec = flatten_dict(frozen_state.to_pure_dict())
    check_shapes({path: spec for path, spec in flat.items() if path not in train_paths}, frozen_spec)
    with ocp.PyTreeCheckpointer() as reader:
        raw_metadata = flatten_dict(reader.metadata(params_path)["params"])
    has_value = all(p[-1] == "value" for p in raw_metadata)
    metadata = {p[:-1] if has_value else p: v for p, v in raw_metadata.items()}
    check_shapes(frozen_spec, metadata)
    report["checkpoint_leaves"] = len(metadata)
    report["checkpoint_dtypes"] = sorted({str(value.dtype) for value in metadata.values()})
    return (graph, state, flat, train_paths, predicate), (frozen_graph, frozen_state), raw_metadata, report


def initialize_adapters(state, flat, train_paths, predicate):
    random_keys = jax.random.split(jax.random.key(SETTINGS.seed), len(train_paths))
    adapters = {}
    for path, key in zip(sorted(train_paths), random_keys, strict=True):
        shape = flat[path].shape
        adapters[path] = (jax.random.normal(key, shape) * 0.01 if path[-1].endswith("_a")
                          else jnp.zeros(shape, dtype=jnp.float32))
    train = state.filter(predicate)
    train.replace_by_pure_dict(unflatten_dict(adapters))
    optimizer = optax.chain(optax.clip_by_global_norm(SETTINGS.gradient_clip),
                            optax.adamw(SETTINGS.learning_rate, weight_decay=SETTINGS.weight_decay))
    opt_state = optimizer.init(train)
    # Adam moments must contain one FP32 pair for each adapter and no base parameters.
    moments = opt_state[1][0]
    for moment in (moments.mu, moments.nu):
        leaf_map = flatten_dict(moment.to_pure_dict())
        if set(leaf_map) != train_paths or any(v.dtype != jnp.float32 for v in leaf_map.values()):
            raise ValueError("Unexpected optimizer state partition or dtype")
    return train, optimizer, opt_state


def load_models(params_path):
    (graph, state, flat, train_paths, predicate), (base_graph, base_state), _, report = checkpoint_layout(params_path)
    restored = openpi_model.restore_params(params_path, dtype=jnp.bfloat16)
    check_shapes(flatten_dict(base_state.to_pure_dict()), flatten_dict(restored))
    base_state.replace_by_pure_dict(restored)
    state.replace_by_pure_dict(restored)
    _, frozen = state.split(predicate, ...)
    train, optimizer, opt_state = initialize_adapters(state, flat, train_paths, predicate)
    report["optimizer_leaves"] = len(jax.tree.leaves(opt_state))
    return (graph, train, frozen, optimizer, opt_state), (base_graph, base_state), report


def cpu_checkpoint_check(params_path, cpu):
    (graph, state, flat, train_paths, predicate), _, metadata, report = checkpoint_layout(params_path)
    # Match the 0.11.13 restore API. An explicit transform tree limits
    # deserialization to the item paths instead of materializing all weights.
    probes = (("action_in_proj", "kernel"), ("action_out_proj", "kernel"),
              ("PaliGemma", "img", "embedding", "kernel"),
              ("PaliGemma", "llm", "final_norm", "scale"),
              ("PaliGemma", "llm", "final_norm_1", "Dense_0", "kernel"))
    where = {p: p if p in metadata else (*p, "value") for p in probes}
    item = {"params": unflatten_dict({p: metadata[p] for p in where.values()})}
    sharding = jax.sharding.SingleDeviceSharding(cpu)
    with ocp.PyTreeCheckpointer() as reader:
        result = reader.restore(params_path, args=ocp.args.PyTreeRestore(
            item=item, transforms={},
            restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(
                sharding=sharding, restore_type=jax.Array, dtype=jnp.bfloat16), item)))
    arrays = flatten_dict(result["params"])
    check_shapes({p: metadata[p] for p in where.values()}, arrays)
    restored = {}
    for view, path in where.items():
        arr = np.asarray(arrays[path])
        if arr.dtype != jnp.bfloat16 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid restored CPU leaf {view}")
        restored["/".join(view)] = hashlib.sha256(arr.tobytes()).hexdigest()
    trained, optimizer, opt_state = initialize_adapters(state, flat, train_paths, predicate)
    _, frozen = state.split(predicate, ...)
    frozen = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.bfloat16), frozen)
    report["cpu_restored_bfloat16_sha256"] = restored
    report["optimizer_leaves"] = len(jax.tree.leaves(opt_state))
    return graph, trained, frozen, optimizer, opt_state, report


class NativePolicy:
    def __init__(self, stats):
        self.adapter = NativeAdapter(stats)
        pace, _ = model_configs()
        self.transform = transforms.compose(openpi_config.ModelTransformFactory()(pace).inputs)

    def prepare(self, samples):
        batch = [self.transform(self.adapter.inputs(sample)) for sample in samples]
        return jax.tree.map(lambda *xs: np.stack(xs), *batch)

    def batch(self, samples):
        batch = self.prepare(samples)
        return openpi_model.Observation.from_dict(batch), batch.get("actions")

    def infer(self, sampler, state, key, native):
        observation, _ = self.batch([native])
        result = np.asarray(sampler(state, key, observation))[0]
        return self.adapter.control_outputs(native["state"], result)


def sampler(graph, *, adapted):
    @jax.jit
    def predict(state, rng, observation):
        model = nnx.merge(graph, *state) if adapted else nnx.merge(graph, state)
        model.eval()
        return model.sample_actions(rng, observation, num_steps=SETTINGS.denoise_steps)
    return predict
