"""Single-GPU ARX LoRA training, paired evaluation, and optional CPU diagnostics."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from . import model
from .adapter import MODEL_CAMERAS
from .config import CODE_ROOT, MANIFEST, OPENPI, PACE_MODEL, SETTINGS, write_json
from .data import ALOHA_SCHEMA, ARM, GRIP, HORIZON, Dataset, digest, partitions
from .sampling import EXECUTED_ROWS, GraspSampler, sample_aloha

EXECUTION_STEPS = 600
EXECUTION_BATCH = 4


def release_digest():
    files = sorted((CODE_ROOT / "scripts/step5_arx").glob("*.py"))
    files += [CODE_ROOT / "scripts/step5_arx/env.sh", CODE_ROOT / "scripts/step5_arx/train.slurm"]
    return hashlib.sha256("".join(digest(path) for path in files).encode()).hexdigest()


def ensure_paths(args):
    root = args.project_root
    if not root.is_absolute() or root.resolve() != root or Path.cwd() != root:
        raise ValueError("Start in the explicit canonical project root")
    if root.parts[-2:] != ("code_cpf", "pi05") or not CODE_ROOT.is_relative_to(root):
        raise ValueError("Expected code_cpf/pi05 with the release below it")
    for name in ("dataset", "data_audit", "aloha_dataset", "aloha_data_audit", "parent_run",
                 "audit_output", "run", "stop_file"):
        path = getattr(args, name, None)
        if path is not None and (not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root)):
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path under the project root")


def platform():
    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError("Expected exactly one logical device")
    return devices[0]


def fingerprint(args):
    record = json.loads(args.data_audit.read_text())
    if record["split"] != partitions() or digest(args.dataset) != record["converted_sha256"]:
        raise ValueError("Converted dataset does not match the audit")
    identity = dict(code_sha256=release_digest(), dataset_sha256=record["converted_sha256"],
                    data_audit_sha256=digest(args.data_audit), settings=SETTINGS.as_dict(),
                    manifest_sha256=digest(MANIFEST), openpi=OPENPI, pace_checkpoint=PACE_MODEL)
    if getattr(args, "aloha_dataset", None) is not None:
        source = json.loads(args.aloha_data_audit.read_text())
        if (source["schema"] != ALOHA_SCHEMA or source["split"] != partitions()
                or digest(args.aloha_dataset) != source["converted_sha256"]):
            raise ValueError("Converted ALOHA dataset does not match the audit")
        identity["aloha"] = dict(dataset_sha256=source["converted_sha256"],
                                 data_audit_sha256=digest(args.aloha_data_audit),
                                 schedule="alternate_two_updates_arx_then_aloha")
    parent = getattr(args, "parent_run", None)
    if parent is not None:
        previous = json.loads((parent / "identity.json").read_text())
        for field in ("dataset_sha256", "aloha", "pace_checkpoint", "openpi"):
            if identity.get(field) != previous.get(field):
                raise ValueError(f"Parent training differs at {field}")
        checkpoint = parent / "checkpoints/step-0400"
        if not complete_checkpoint(checkpoint):
            raise ValueError("Expected the completed mixed step-0400")
        identity["execution_objective"] = dict(
            parent_identity_sha256=digest(parent / "identity.json"),
            parent_checkpoint_metadata_sha256=digest(checkpoint / "_CHECKPOINT_METADATA"),
            near_rows=EXECUTED_ROWS, near_loss_mass=0.5, terminal_windows=True)
        identity["settings"].update(train_steps=EXECUTION_STEPS, batch_size=EXECUTION_BATCH)
    return identity


def optimizer_step(graph, optimizer):
    @jax.jit
    def step(train, frozen, opt_state, key, obs, target, valid_rows=None):
        def loss(adapters):
            pi = nnx.merge(graph, adapters, frozen)
            pi.train()
            losses = pi.compute_loss(key, obs, target, train=True)
            if valid_rows is None:
                return jnp.mean(losses)
            row = jnp.arange(HORIZON)
            weights = jnp.where(row < EXECUTED_ROWS, 1., EXECUTED_ROWS / (HORIZON - EXECUTED_ROWS))
            weights = weights * (row < valid_rows[:, None])
            return jnp.mean(jnp.sum(losses * weights, axis=-1) / jnp.sum(weights, axis=-1))
        value, grad = jax.value_and_grad(loss)(train)
        updates, opt_state = optimizer.update(grad, opt_state, train)
        updated = optax.apply_updates(train, updates)
        difference = jax.tree.map(lambda new, old: new - old, updated, train)
        return updated, opt_state, (value, optax.global_norm(grad), optax.global_norm(difference))
    return step


def flow_loss(graph, *, adapted):
    @jax.jit
    def loss(state, key, obs, target):
        pi = nnx.merge(graph, *state) if adapted else nnx.merge(graph, state)
        pi.eval()
        return jnp.mean(pi.compute_loss(key, obs, target, train=False))
    return loss


def lower_initial_loss(loss, initial, frozen, rng, observation, target):
    # JAX's AOT metadata reconstructs argument trees with ArgInfo leaves.
    # Pass flat arrays at that boundary and rebuild the typed Observation in tracing.
    leaves, structure = jax.tree.flatten(observation)

    @jax.jit
    def invoke(state, key, obs_leaves, actions):
        return loss(state, key, jax.tree.unflatten(structure, obs_leaves), actions)

    return invoke.lower((initial, frozen), rng, leaves, target), leaves


def check_initial_loss(loss, vanilla_loss, initial, frozen, base_state, rng, observation, target, path):
    """Record the optional initialization comparison for a fixed batch."""
    lowered, leaves = lower_initial_loss(loss, initial, frozen, rng, observation, target)
    compiled = lowered.compile()

    def evaluate_adapters(adapters):
        return float(compiled((adapters, frozen), rng, leaves, target))

    seeded = evaluate_adapters(initial)
    zeroed = evaluate_adapters(jax.tree.map(jnp.zeros_like, initial))
    vanilla = float(vanilla_loss(base_state, rng, observation, target))
    rtol, atol = 1e-3, 1e-5
    finite = bool(np.isfinite([seeded, zeroed, vanilla]).all())
    same_diff = abs(seeded - zeroed)
    limit = atol + rtol * abs(zeroed)
    passed = finite and same_diff <= limit

    def json_number(value):
        return value if np.isfinite(value) else str(value)

    write_json(path, dict(method="same_compiled_lora_executable_with_dynamic_seeded_vs_zero_adapters",
                          status="passed" if passed else "failed",
                          seeded_lora=json_number(seeded), zero_lora=json_number(zeroed),
                          vanilla_frozen=json_number(vanilla), vanilla_role="diagnostic_only",
                          same_graph_abs_diff=json_number(same_diff),
                          same_graph_rel_diff=json_number(same_diff / max(abs(zeroed), 1e-12)),
                          vanilla_abs_diff=json_number(abs(seeded - vanilla)),
                          vanilla_rel_diff=json_number(abs(seeded - vanilla) / max(abs(vanilla), 1e-12)),
                          all_losses_finite=finite, rtol=rtol, atol=atol, same_graph_abs_limit=json_number(limit)))
    if not finite:
        raise RuntimeError(f"Nonfinite initialization losses: seeded={seeded}, zero={zeroed}, vanilla={vanilla}")
    return seeded, evaluate_adapters


def update(step, train, frozen, optimizer_state, key, observation, target, valid_rows=None):
    fitted, optimizer_state, values = step(train, frozen, optimizer_state, key, observation, target, valid_rows)
    loss, grad, change = map(float, values)
    if not np.isfinite([loss, grad, change]).all():
        raise RuntimeError(f"Invalid optimizer update: loss={loss}, grad={grad}, change={change}")
    metrics = dict(loss=loss, grad_norm=grad, adapter_delta_norm=change)
    if grad == 0 or change == 0:
        print(json.dumps(dict(warning="zero_gradient_or_adapter_update", **metrics)), flush=True)
    return fitted, optimizer_state, metrics


def fitted_state(train, opt):
    return {"lora": train.to_pure_dict(), "optimizer": jax.tree.leaves(opt)}


def complete_checkpoint(path):
    if not path.is_dir() or not (path / "_METADATA").is_file() or not ocp.utils.is_checkpoint_finalized(path):
        return False
    try:
        metadata = json.loads((path / "_CHECKPOINT_METADATA").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return bool(metadata.get("commit_timestamp_nsecs"))


def latest_checkpoint(directory):
    paths = sorted(directory.glob("step-[0-9][0-9][0-9][0-9]"), reverse=True)
    return next((path for path in paths if complete_checkpoint(path)), None)


def restore(path, train, opt):
    if not complete_checkpoint(path):
        raise ValueError(f"Checkpoint was not committed: {path}")
    target = fitted_state(train, opt)
    with ocp.StandardCheckpointer() as store:
        saved = store.restore(path, target=target, strict=True)
    for before, after in zip(jax.tree.leaves(target), jax.tree.leaves(saved), strict=True):
        if (before.shape, before.dtype) != (after.shape, after.dtype):
            raise ValueError(f"Saved state shape or dtype mismatch at {path}")
    graph_opt = jax.tree.structure(opt)
    opt = jax.tree.unflatten(graph_opt, saved["optimizer"])
    if int(opt[1][0].count) != int(path.name.split("-")[-1]):
        raise ValueError(f"AdamW update count disagrees with checkpoint step: {path}")
    train = jax.tree.map(lambda value: value, train)
    train.replace_by_pure_dict(saved["lora"])
    return train, opt


def save(path, train, opt):
    if path.exists():
        if complete_checkpoint(path):
            raise FileExistsError(f"Refusing to replace committed checkpoint: {path}")
        path.rename(path.with_name(f".incomplete-{path.name}-{time.time_ns()}"))
    with ocp.StandardCheckpointer() as store:
        store.save(path, fitted_state(train, opt))
        store.wait_until_finished()
    if not complete_checkpoint(path):
        raise RuntimeError(f"Orbax checkpoint did not finalize: {path}")


def evaluate(dataset, split, policy, predict, loss_fn, state, *, stopped=lambda: False, windows=None,
             terminal=False):
    rows = []
    windows = dataset.indices(split, SETTINGS.evaluation_stride) if windows is None else windows
    for ep, t in windows:
        if stopped():
            raise InterruptedError("Stop requested during validation")
        sample = dataset.read(split, ep, t, terminal=terminal)
        obs, target = policy.batch([sample])
        rng = jax.random.fold_in(jax.random.key(SETTINGS.seed + 1), ep * 10000 + t)
        flow_key, sample_key = jax.random.split(rng)
        loss = float(loss_fn(state, flow_key, obs, target))
        pred = policy.adapter.outputs(sample["state"], np.asarray(predict(state, sample_key, obs))[0])
        valid = sample.get("valid_rows", HORIZON)
        diff = np.abs(pred[:valid] - sample["actions"][:valid])
        metrics = dict(flow_loss=loss, arm_mae_rad=float(diff[:, ARM].mean()),
                       k4_left_arm_mae_rad=float(diff[:EXECUTED_ROWS, :6].mean()),
                       k4_left_gripper_mae=float(diff[:EXECUTED_ROWS, 6].mean()),
                       k4_right_arm_mae_rad=float(diff[:EXECUTED_ROWS, 7:13].mean()),
                       k4_right_gripper_mae=float(diff[:EXECUTED_ROWS, 13].mean()),
                       gripper_mae=float(diff[:, GRIP].mean()),
                       first_arm_mae_rad=float(diff[0, ARM].mean()),
                       first_gripper_mae=float(diff[0, GRIP].mean()),
                       gripper_out_of_range_rate=float(np.mean((pred[:, GRIP] < 0) | (pred[:, GRIP] > 1))))
        if not np.isfinite(list(metrics.values())).all():
            raise RuntimeError(f"Invalid evaluation output for episode {ep}, row {t}")
        rows.append(dict(episode=ep, t=t, **metrics))
    episodes = sorted({row["episode"] for row in rows})
    per_episode = {ep: {k: float(np.mean([row[k] for row in rows if row["episode"] == ep]))
                        for k in metrics} for ep in episodes}
    average = {k: float(np.mean([item[k] for item in per_episode.values()])) for k in metrics}
    return dict(split=split, windows=len(rows), episode_mean=average, episodes=per_episode, rows=rows)


def versions():
    return {p: importlib.metadata.version(p) for p in
            ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "optax",
             "orbax-checkpoint", "h5py", "numpy", "pillow")}


def arx_input_contract(obs, target, key):
    expected_cameras = set(MODEL_CAMERAS)
    if type(obs) is not model.openpi_model.Observation:
        raise TypeError(f"Expected OpenPI Observation, got {type(obs)}")
    if set(obs.images) != expected_cameras or set(obs.image_masks) != expected_cameras:
        raise ValueError("Unexpected ARX camera keys")
    expected = [(image, (1, 224, 224, 3), np.float32) for image in obs.images.values()]
    expected += [(mask, (1,), np.bool_) for mask in obs.image_masks.values()]
    expected += [(obs.state, (1, 32), np.float32), (obs.tokenized_prompt, (1, 200), np.int32),
                 (obs.tokenized_prompt_mask, (1, 200), np.bool_), (target, (1, 50, 32), np.float32)]
    for array, shape, dtype in expected:
        if (array is None or array.shape != shape or jax.dtypes.canonicalize_dtype(array.dtype) != dtype
                or not np.isfinite(array).all()):
            actual = None if array is None else (array.shape, array.dtype)
            raise ValueError(f"Unexpected ARX input {actual}; expected {shape}, JAX {dtype}")
    count = int(obs.tokenized_prompt_mask.sum())
    if (obs.token_ar_mask is not None or obs.token_loss_mask is not None or np.any(target[..., 14:])
            or not all(np.all(mask) for mask in obs.image_masks.values()) or not 0 < count < 200
            or key.shape != () or not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)):
        raise ValueError("Unexpected ARX masking, action padding or PRNG key")
    leaves, structure = jax.tree_util.tree_flatten_with_path(obs)
    return dict(observation=f"{type(obs).__module__}.{type(obs).__qualname__}",
                pytree=str(structure), leaves=[dict(path=jax.tree_util.keystr(path), shape=list(value.shape),
                dtype=str(value.dtype), jax_dtype=str(jax.dtypes.canonicalize_dtype(value.dtype)))
                                               for path, value in leaves],
                actions=dict(shape=list(target.shape), dtype=str(target.dtype)),
                prng=dict(shape=list(key.shape), dtype=str(key.dtype), data=jax.random.key_data(key).tolist()),
                valid_prompt_tokens=count)


def audit_cpu(args):
    if platform().platform != "cpu":
        raise RuntimeError("The optional CPU audit requires a CPU-only JAX device")
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    snapshot = model.checkpoint_snapshot(args.project_root)
    params = model.validate_install(args.project_root, full_hash=True)
    identity = fingerprint(args)
    print("CPU manifest and dataset verified", flush=True)
    graph, train, frozen, optimizer, opt, baseline, report = model.cpu_checkpoint_check(params, platform())
    jax.block_until_ready((train, opt))
    print("CPU graph, 5 Orbax tensor reads and optimizer initialized", flush=True)
    with Dataset(args.dataset) as dataset:
        policy = model.NativePolicy(dataset.norm_stats)
        episode, row = GraspSampler(dataset).window(SETTINGS.seed, 4)
        sample = dataset.read("train", episode, row)
        obs, target = policy.batch([sample])
        key = jax.random.key(SETTINGS.seed)
        input_report = arx_input_contract(obs, target, key)
        input_report.update(episode=episode, row=row)
        write_json(args.audit_output.with_name("arx-input-contract.json"), input_report)
        np.testing.assert_allclose(policy.adapter.outputs(sample["state"], target[0]), sample["actions"],
                                   rtol=0, atol=5e-7)
        lowered, _ = lower_initial_loss(flow_loss(graph, adapted=True), train, frozen, key, obs, target)
        expected_loss = ((), jnp.float32)
        if (lowered.out_info.shape, lowered.out_info.dtype) != expected_loss:
            raise RuntimeError("Unexpected abstract LoRA evaluation loss")
        del lowered
        print("Full adapted initialization loss lowered on the real ARX Observation (no executable run)", flush=True)
        base_graph, base_state = baseline
        base_loss = jax.eval_shape(flow_loss(base_graph, adapted=False), base_state, key, obs, target)
        prediction = jax.eval_shape(model.sampler(graph, adapted=True), (train, frozen), key, obs)
        base_prediction = jax.eval_shape(model.sampler(base_graph, adapted=False), base_state, key, obs)
        if (base_loss.shape, base_loss.dtype) != expected_loss or any(
                (value.shape, value.dtype) != (target.shape, target.dtype)
                for value in (prediction, base_prediction)):
            raise RuntimeError("Unexpected abstract frozen loss or sampler output")
        traced = jax.eval_shape(optimizer_step(graph, optimizer), train, frozen, opt, key, obs, target)
        expected = (train, opt)
        if (jax.tree.structure(traced[:2]) != jax.tree.structure(expected)
                or any((actual.shape, actual.dtype) != (prior.shape, prior.dtype)
                       for actual, prior in zip(jax.tree.leaves(traced[:2]), jax.tree.leaves(expected), strict=True))
                or any((value.shape, value.dtype) != expected_loss for value in traced[2])):
            raise RuntimeError("Abstract optimizer step changed its state contract")
        print("Full train=True gradient/Optax and eval/sampling contracts traced (no model kernels executed)", flush=True)
    # Exercise the same serializer and restore tree used by interrupted formal training.
    import tempfile
    with tempfile.TemporaryDirectory(dir=args.audit_output.parent) as temp:
        directory = Path(temp)
        (directory / "step-0000").mkdir()  # An interrupted numeric directory is quarantined before saving.
        test = directory / "step-0000"
        save(test, train, opt)
        (directory / "step-0001").mkdir()
        (directory / "step-0002.orbax-checkpoint-tmp-0").mkdir()
        if latest_checkpoint(directory) != test:
            raise RuntimeError("Checkpoint recovery did not skip interrupted writes")
        reloaded = restore(test, train, opt)
        if not all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves((train, opt)),
                                                        jax.tree.leaves(reloaded), strict=True)):
            raise RuntimeError("Orbax adapter/optimizer round trip changed values")
    if snapshot != model.checkpoint_snapshot(args.project_root):
        raise RuntimeError("Model files changed during CPU audit")
    print("Committed checkpoint restored; incomplete paths skipped", flush=True)
    write_json(args.audit_output, dict(status="CPU_STRUCTURAL_ONLY", identity=identity, graph=report, versions=versions(),
                                     checkpoint_snapshot=snapshot, checkpoint_recovery_checked=True,
                                     model_input_shape=list(target.shape), abstract_update_traced=True,
                                     token_count=input_report["valid_prompt_tokens"], input_contract=input_report,
                                     full_initial_eval_lowered=True, full_eval_and_sampling_traced=True))


def gpu_device():
    cuda = model.cuda_identity()
    print(json.dumps(dict(stage="cuda_device", **cuda)), flush=True)
    device = platform()
    if device.platform != "gpu":
        raise RuntimeError(f"Expected one CUDA device, found {device.device_kind}")
    print(json.dumps(dict(gpu_device=str(device), gpu_model=device.device_kind,
                          memory=device.memory_stats())), flush=True)
    return device


def stop_requested(args):
    return args.stop_file is not None and args.stop_file.stat().st_size > 0


def record_gpu_progress(run, stage, **details):
    event = dict(stage=stage, **details)
    write_json(run / "gpu_progress.json", event)
    print(json.dumps(event), flush=True)


def write_gpu_failure(run, error):
    failure = dict(stage="gpu_fit_failed", error_type=type(error).__name__, error=str(error))
    try:
        failure["memory"] = platform().memory_stats()
    except Exception as diagnostic:
        failure["memory_error"] = str(diagnostic)
    try:
        progress = run / "gpu_progress.json"
        failure["last_progress"] = json.loads(progress.read_text()) if progress.exists() else None
        checkpoint = latest_checkpoint(run / "checkpoints")
        failure["last_committed_checkpoint"] = str(checkpoint) if checkpoint else None
        write_json(run / "gpu_failure.json", failure)
    except Exception as diagnostic:
        failure["record_error"] = str(diagnostic)
    print(json.dumps(failure), flush=True)


def run_fit(args, identity):
    run = args.run
    execution_fit = args.parent_run is not None
    train_steps = identity["settings"]["train_steps"]
    stopping = False

    def stop_after_save(*_):
        nonlocal stopping
        stopping = True

    def stopped():
        return stopping or stop_requested(args)

    signal.signal(signal.SIGUSR1, stop_after_save)
    signal.signal(signal.SIGTERM, stop_after_save)
    dir_ckpt = run / "checkpoints"
    if not args.resume:
        write_json(run / "identity.json", identity)
    elif json.loads((run / "identity.json").read_text()) != identity:
        raise RuntimeError("Cannot resume with a different dataset, code or config")
    dir_ckpt.mkdir(exist_ok=True)
    if stopped():
        return 75
    gpu = gpu_device()
    record_gpu_progress(run, "gpu_weights_verify", status="running")
    params = model.validate_install(args.project_root, full_hash=True)
    record_gpu_progress(run, "gpu_checkpoint_restore", status="running")
    parts, _, report = model.load_models(params)
    graph, initial, frozen, optimizer, initial_opt = parts
    step = optimizer_step(graph, optimizer)
    loss = flow_loss(graph, adapted=True)
    pred = model.sampler(graph, adapted=True)
    rng = jax.random.key(SETTINGS.seed)
    write_json(run / "parameter_layout.json", report)
    record_gpu_progress(run, "gpu_checkpoint_restore", status="passed", memory=gpu.memory_stats())
    with ExitStack() as datasets:
        dataset = datasets.enter_context(Dataset(args.dataset))
        policy = model.NativePolicy(dataset.norm_stats)
        sampler = GraspSampler(dataset)
        aloha_dataset = aloha_policy = aloha_sampler = None
        if args.aloha_dataset is not None:
            aloha_dataset = datasets.enter_context(Dataset(args.aloha_dataset, schema=ALOHA_SCHEMA))
            aloha_policy = model.NativePolicy(aloha_dataset.norm_stats)
            aloha_sampler = GraspSampler(aloha_dataset)
        validation_phases = GraspSampler(dataset, "val").phases
        release_windows = []
        if execution_fit:
            terminal_pools = {sampler: tuple(dataset.indices("train", terminal=True)),
                              aloha_sampler: tuple(aloha_dataset.indices("train", terminal=True))}
            for ep, group in sorted(dataset.groups["val"].items()):
                right = group["drive_target"][:, 13]
                first_open = int(np.flatnonzero(np.diff(right) > 1e-4)[0]) + 1
                release_windows.append((ep, first_open - 2))
        last = latest_checkpoint(dir_ckpt)
        start = int(last.name.split("-")[-1]) if last else 0
        fitted, opt = restore(last, initial, initial_opt) if last else (initial, initial_opt)
        if execution_fit and last is None:
            fitted, _ = restore(args.parent_run / "checkpoints/step-0400", initial, initial_opt)
        for i in range(start, train_steps + 1):
            if i > start:
                record_gpu_progress(run, "gpu_formal_train", step=i, status="running")
                train_dataset, train_policy, train_sampler = dataset, policy, sampler
                training_body = "arx-x5"
                if aloha_dataset is not None and sample_aloha(i):
                    train_dataset, train_policy, train_sampler = aloha_dataset, aloha_policy, aloha_sampler
                    training_body = "aloha-agilex"
                if execution_fit:
                    windows = train_sampler.batch(SETTINGS.seed, i, EXECUTION_BATCH, terminal_pools[train_sampler])
                else:
                    windows = [train_sampler.window(SETTINGS.seed, i)]
                samples = [train_dataset.read("train", ep, row, terminal=execution_fit) for ep, row in windows]
                obs, target = train_policy.batch(samples)
                valid = np.asarray([s["valid_rows"] for s in samples], dtype=np.int32) if execution_fit else None
                fitted, opt, metrics = update(step, fitted, frozen, opt, jax.random.fold_in(rng, i),
                                             obs, target, valid)
                count = int(opt[1][0].count)
                if count != i:
                    raise RuntimeError(f"Formal AdamW count {count} does not match step {i}")
                episode, row = windows[0]
                metrics.update(adamw_count=count, embodiment=training_body, episode=episode, row=row)
                if execution_fit:
                    metrics["windows"] = windows
                    metrics["valid_rows"] = valid.tolist()
                if i <= 2:
                    record_gpu_progress(run, "gpu_formal_train", step=i, status="passed",
                                        memory=gpu.memory_stats(), **metrics)
                else:
                    print(json.dumps(dict(step=i, **metrics)), flush=True)
            checkpoint_interval = SETTINGS.validation_interval if execution_fit else SETTINGS.checkpoint_interval
            if i and (i == 2 or i % checkpoint_interval == 0 or stopped() or i == train_steps):
                checkpoint = dir_ckpt / f"step-{i:04d}"
                if not complete_checkpoint(checkpoint):
                    record_gpu_progress(run, "gpu_checkpoint_save", step=i, status="running")
                    save(checkpoint, fitted, opt)
                    record_gpu_progress(run, "gpu_checkpoint_save", step=i, status="passed",
                                        checkpoint=str(checkpoint), memory=gpu.memory_stats())
            if stopped():
                return 75
            if (execution_fit or i) and (i % SETTINGS.validation_interval == 0 or i == train_steps):
                out = run / f"val-{i:04d}.json"
                if not out.exists():
                    record_gpu_progress(run, "gpu_validation", step=i, status="running")
                    report = evaluate(dataset, "val", policy, pred, loss, (fitted, frozen), stopped=stopped)
                    for name, pool in validation_phases.items():
                        # Use one central window from each validation episode.
                        anchors = pool[2::EXECUTED_ROWS]
                        phase = evaluate(dataset, "val", policy, pred, loss, (fitted, frozen),
                                         stopped=stopped, windows=anchors)
                        report.setdefault("grasp_windows", {})[name] = phase
                    if execution_fit:
                        report["release_windows"] = evaluate(dataset, "val", policy, pred, loss, (fitted, frozen),
                                                             stopped=stopped, windows=release_windows, terminal=True)
                        phases = report["grasp_windows"]
                        report["episode_mean"]["execution_score"] = float(np.mean([
                            phases["approach"]["episode_mean"]["k4_left_arm_mae_rad"],
                            phases["close"]["episode_mean"]["k4_left_gripper_mae"],
                            phases["post_close"]["episode_mean"]["k4_left_arm_mae_rad"],
                            report["release_windows"]["episode_mean"]["k4_right_gripper_mae"],
                            report["episode_mean"]["k4_right_arm_mae_rad"],
                        ]))
                    write_json(out, report)
        reports = [(int(path.stem.split("-")[-1]), json.loads(path.read_text())["episode_mean"])
                   for path in run.glob("val-[0-9][0-9][0-9][0-9].json") if path.stem != "val-0000"]
        leading_metric = "execution_score" if execution_fit else "flow_loss"
        best, metrics = min(reports, key=lambda row: (row[1][leading_metric], row[1]["arm_mae_rad"],
                                                      row[1]["gripper_mae"], row[0]))
        write_json(run / "selected.json", dict(step=best, metrics=metrics, last_step=train_steps,
                                               memory=gpu.memory_stats()))
        record_gpu_progress(run, "gpu_fit_complete", step=train_steps, memory=gpu.memory_stats())
    return 0


def test_selected(args, identity):
    if json.loads((args.run / "identity.json").read_text()) != identity:
        raise RuntimeError("Test dataset or release differs from training")
    chosen = json.loads((args.run / "selected.json").read_text())["step"]
    path = args.run / "checkpoints" / f"step-{chosen:04d}"
    gpu_device()
    parts, baseline, _ = model.load_models(model.validate_install(args.project_root, full_hash=True))
    graph, initial, frozen, _, opt = parts
    base_graph, base_state = baseline
    fitted, _ = restore(path, initial, opt)
    with Dataset(args.dataset) as dataset:
        policy = model.NativePolicy(dataset.norm_stats)
        out = args.run / "test-paired.json"
        if out.exists():
            raise FileExistsError(out)
        write_json(out, dict(selected_step=chosen,
                            frozen=evaluate(dataset, "test", policy, model.sampler(base_graph, adapted=False),
                                            flow_loss(base_graph, adapted=False), base_state),
                            lora=evaluate(dataset, "test", policy, model.sampler(graph, adapted=True),
                                          flow_loss(graph, adapted=True), (fitted, frozen))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "test", "audit-cpu"))
    for name in ("project-root", "dataset", "data-audit"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--aloha-dataset", type=Path)
    parser.add_argument("--aloha-data-audit", type=Path)
    parser.add_argument("--parent-run", type=Path, help="Warm-start the execution-aligned mixed experiment")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--audit-output", "--cpu-receipt", dest="audit_output", type=Path,
                        help="Output for the optional audit-cpu command")
    args = parser.parse_args()
    if (args.aloha_dataset is None) != (args.aloha_data_audit is None):
        parser.error("--aloha-dataset and --aloha-data-audit must be supplied together")
    if args.parent_run is not None and (args.aloha_dataset is None or args.command == "audit-cpu"):
        parser.error("--parent-run requires the mixed dataset for fit or test")
    ensure_paths(args)
    if args.command == "audit-cpu":
        if args.audit_output is None:
            parser.error("audit-cpu requires --audit-output")
        audit_cpu(args)
        return
    if args.run is None:
        parser.error("fit and test require --run")
    identity = fingerprint(args)
    if args.command == "fit" and not args.resume:
        args.run.mkdir()
    with (args.run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "test":
            test_selected(args, identity)
            return
        try:
            raise SystemExit(run_fit(args, identity))
        except InterruptedError:
            raise SystemExit(75) from None
        except Exception as error:
            write_gpu_failure(args.run, error)
            raise


if __name__ == "__main__":
    main()
