"""CPU checkpoint audit, gated single-GPU fit, and paired held-out evaluation."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from . import model
from .config import CODE_ROOT, MANIFEST, OPENPI, PACE_MODEL, SETTINGS, write_json
from .data import ARM, GRIP, Dataset, digest, partitions

MIN_DISK_BYTES = 16 * 2**30
MIN_VRAM_BYTES = 44 * 2**30


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
    for name in ("dataset", "data_audit", "cpu_receipt", "run", "stop_file"):
        path = getattr(args, name, None)
        if path is not None and (not path.is_absolute() or not path.resolve().is_relative_to(root)):
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
    return dict(code_sha256=release_digest(), dataset_sha256=record["converted_sha256"],
                data_audit_sha256=digest(args.data_audit), settings=SETTINGS.as_dict(),
                manifest_sha256=digest(MANIFEST), openpi=OPENPI, pace_checkpoint=PACE_MODEL)


def optimizer_step(graph, optimizer):
    @jax.jit
    def step(train, frozen, opt_state, key, obs, target):
        def loss(adapters):
            pi = nnx.merge(graph, adapters, frozen)
            pi.train()
            return jnp.mean(pi.compute_loss(key, obs, target, train=True))
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


def update(step, train, frozen, optimizer_state, key, observation, target):
    fitted, optimizer_state, values = step(train, frozen, optimizer_state, key, observation, target)
    loss, grad, change = map(float, values)
    if not np.isfinite([loss, grad, change]).all() or grad <= 0 or change <= 0:
        raise RuntimeError(f"Invalid optimizer update: loss={loss}, grad={grad}, change={change}")
    return fitted, optimizer_state, dict(loss=loss, grad_norm=grad, adapter_delta_norm=change)


def fitted_state(train, opt):
    return {"lora": train.to_pure_dict(), "optimizer": jax.tree.leaves(opt)}


def restore(path, train, opt):
    target = fitted_state(train, opt)
    with ocp.StandardCheckpointer() as store:
        saved = store.restore(path, target=target, strict=True)
    for before, after in zip(jax.tree.leaves(target), jax.tree.leaves(saved), strict=True):
        if (before.shape, before.dtype) != (after.shape, after.dtype):
            raise ValueError(f"Saved state shape or dtype mismatch at {path}")
    graph_opt = jax.tree.structure(opt)
    opt = jax.tree.unflatten(graph_opt, saved["optimizer"])
    train = jax.tree.map(lambda value: value, train)
    train.replace_by_pure_dict(saved["lora"])
    return train, opt


def save(path, train, opt):
    with ocp.StandardCheckpointer() as store:
        store.save(path, fitted_state(train, opt))
        store.wait_until_finished()


def evaluate(dataset, split, policy, predict, loss_fn, state, *, stopped=lambda: False):
    rows = []
    for ep, t in dataset.indices(split, SETTINGS.evaluation_stride):
        if stopped():
            raise InterruptedError("Stop requested during validation")
        sample = dataset.read(split, ep, t)
        obs, target = policy.batch([sample])
        rng = jax.random.fold_in(jax.random.key(SETTINGS.seed + 1), ep * 10000 + t)
        flow_key, sample_key = jax.random.split(rng)
        loss = float(loss_fn(state, flow_key, obs, target))
        pred = policy.adapter.outputs(sample["state"], np.asarray(predict(state, sample_key, obs))[0])
        diff = np.abs(pred - sample["actions"])
        metrics = dict(flow_loss=loss, arm_mae_rad=float(diff[:, ARM].mean()),
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
            ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "h5py", "numpy", "pillow")}


def audit_cpu(args):
    if platform().platform != "cpu":
        raise RuntimeError("CPU receipt requires a CPU-only JAX device")
    if args.cpu_receipt.exists():
        raise FileExistsError(args.cpu_receipt)
    params = model.validate_install(args.project_root, full_hash=True)
    identity = fingerprint(args)
    print("CPU manifest and dataset verified", flush=True)
    graph, train, frozen, optimizer, opt, report = model.cpu_checkpoint_check(params, platform())
    jax.block_until_ready((train, opt))
    print("CPU graph, 5 Orbax tensor reads and optimizer initialized", flush=True)
    with Dataset(args.dataset) as dataset:
        policy = model.NativePolicy(dataset.norm_stats)
        sample = dataset.read("train", *next(dataset.indices("train")))
        obs, target = policy.batch([sample])
        if obs.tokenized_prompt.shape != (1, 200) or not 0 < obs.tokenized_prompt_mask.sum() < 200:
            raise RuntimeError("Unexpected token shape, truncation, or empty prompt")
        if target.shape != (1, 50, 32) or np.any(target[..., 14:]):
            raise RuntimeError("Wrong model action padding")
        if not all(img.shape == (1, 224, 224, 3) for img in obs.images.values()):
            raise RuntimeError("Wrong resized image shape")
        np.testing.assert_allclose(policy.adapter.outputs(sample["state"], target[0]), sample["actions"],
                                   rtol=0, atol=5e-7)
        traced = jax.eval_shape(optimizer_step(graph, optimizer), train, frozen, opt,
                                jax.random.key(SETTINGS.seed), obs, target)
        if jax.tree.structure(traced[:2]) != jax.tree.structure((train, opt)):
            raise RuntimeError("Abstract optimizer step changed its state structure")
        print("Production gradient/update traced on CPU without executing model kernels", flush=True)
    # Exercise the same serializer and restore tree used by interrupted formal training.
    import tempfile
    with tempfile.TemporaryDirectory(dir=args.cpu_receipt.parent) as temp:
        test = Path(temp) / "adapter"
        save(test, train, opt)
        reloaded = restore(test, train, opt)
        if not all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves((train, opt)),
                                                        jax.tree.leaves(reloaded), strict=True)):
            raise RuntimeError("Orbax adapter/optimizer round trip changed values")
    write_json(args.cpu_receipt, dict(status="cpu_pass", identity=identity, graph=report, versions=versions(),
                                     model_input_shape=list(target.shape), abstract_update_traced=True,
                                     token_count=int(obs.tokenized_prompt_mask.sum())))


def admit_gpu(args):
    device = platform()
    memory = device.memory_stats() or {}
    supported = memory.get("bytes_limit", 0)
    if device.platform != "gpu" or supported < MIN_VRAM_BYTES:
        raise RuntimeError(f"Expected one CUDA device with at least {MIN_VRAM_BYTES} usable bytes; {memory}")
    if shutil.disk_usage(args.run.parent).free < MIN_DISK_BYTES:
        raise RuntimeError("Need 16 GiB free on the run filesystem")
    approved = json.loads(args.cpu_receipt.read_text())
    identity = fingerprint(args)
    if approved["status"] != "cpu_pass" or approved["identity"] != identity or approved["versions"] != versions():
        raise RuntimeError("CPU receipt is missing or stale for this code, dataset or runtime")
    print(json.dumps(dict(gpu_device=str(device), gpu_model=device.device_kind, memory=memory)), flush=True)
    return identity


def stop_requested(args):
    return args.stop_file is not None and args.stop_file.stat().st_size > 0


def run_fit(args, identity):
    run = args.run
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
    params = model.validate_install(args.project_root)
    parts, baseline, report = model.load_models(params)
    graph, initial, frozen, optimizer, initial_opt = parts
    base_graph, base_state = baseline
    step = optimizer_step(graph, optimizer)
    loss = flow_loss(graph, adapted=True)
    pred = model.sampler(graph, adapted=True)
    rng = jax.random.key(SETTINGS.seed)
    write_json(run / "parameter_layout.json", report)
    with Dataset(args.dataset) as dataset:
        policy = model.NativePolicy(dataset.norm_stats)
        indices = tuple(dataset.indices("train"))
        overfit = [dataset.read("train", *indices[len(indices) // 2])]
        obs, target = policy.batch(overfit)
        smoke = run / "gpu_gates.json"
        if not smoke.exists():
            first_val = float(loss((initial, frozen), rng, obs, target))
            ref = np.asarray(flow_loss(base_graph, adapted=False)(base_state, rng, obs, target))
            if not np.isfinite([first_val, ref.item()]).all():
                raise RuntimeError("Nonfinite frozen/LoRA initialization loss")
            np.testing.assert_allclose(first_val, ref, rtol=1e-3, atol=1e-5)
            _, _, smoke_result = update(step, initial, frozen, initial_opt, rng, obs, target)
            fitted, opt = initial, initial_opt
            overfit_log = []
            for i in range(SETTINGS.overfit_steps):
                fitted, opt, metrics = update(step, fitted, frozen, opt, jax.random.fold_in(rng, i), obs, target)
                overfit_log.append(metrics)
                if stopped():
                    return 75
            last_val = float(loss((fitted, frozen), rng, obs, target))
            relative = (first_val - last_val) / max(first_val, 1e-12)
            gates = dict(smoke=smoke_result, overfit=overfit_log, initial_eval_loss=first_val,
                         final_eval_loss=last_val, relative_improvement=relative)
            if not np.isfinite(last_val) or relative < SETTINGS.overfit_min_relative_improvement:
                write_json(run / "gpu_gates_failed.json", gates)
                raise RuntimeError(f"20-step overfit gate failed: relative reduction {relative:.4g}")
            write_json(smoke, dict(status="passed", **gates))
        else:
            if json.loads(smoke.read_text())["status"] != "passed":
                raise RuntimeError("GPU gate receipt is invalid")
        if stopped():
            return 75
        val_baseline = run / "val-frozen.json"
        if not val_baseline.exists():
            write_json(val_baseline, evaluate(dataset, "val", policy, model.sampler(base_graph, adapted=False),
                                             flow_loss(base_graph, adapted=False), base_state, stopped=stopped))
        checkpoints = sorted(p for p in dir_ckpt.glob("step-[0-9][0-9][0-9][0-9]") if p.is_dir())
        start = int(checkpoints[-1].name.split("-")[-1]) if checkpoints else 0
        fitted, opt = restore(checkpoints[-1], initial, initial_opt) if checkpoints else (initial, initial_opt)
        for i in range(start, SETTINGS.train_steps + 1):
            if i > start:
                index = np.random.default_rng(np.random.SeedSequence([SETTINGS.seed, i])).integers(len(indices))
                obs, target = policy.batch([dataset.read("train", *indices[index])])
                fitted, opt, metrics = update(step, fitted, frozen, opt, jax.random.fold_in(rng, i), obs, target)
                print(json.dumps(dict(step=i, **metrics)), flush=True)
            if i and (i % SETTINGS.checkpoint_interval == 0 or stopped() or i == SETTINGS.train_steps):
                checkpoint = dir_ckpt / f"step-{i:04d}"
                if not checkpoint.exists():
                    save(checkpoint, fitted, opt)
            if stopped():
                return 75
            if i and (i % SETTINGS.validation_interval == 0 or i == SETTINGS.train_steps):
                out = run / f"val-{i:04d}.json"
                if not out.exists():
                    write_json(out, evaluate(dataset, "val", policy, pred, loss, (fitted, frozen), stopped=stopped))
        reports = [(int(path.stem.split("-")[-1]), json.loads(path.read_text())["episode_mean"])
                   for path in run.glob("val-[0-9][0-9][0-9][0-9].json")]
        best, metrics = min(reports, key=lambda row: (row[1]["flow_loss"], row[1]["arm_mae_rad"],
                                                      row[1]["gripper_mae"], row[0]))
        write_json(run / "selected.json", dict(step=best, metrics=metrics, last_step=SETTINGS.train_steps))
    return 0


def test_selected(args, identity):
    if json.loads((args.run / "identity.json").read_text()) != identity:
        raise RuntimeError("Test dataset or release differs from training")
    chosen = json.loads((args.run / "selected.json").read_text())["step"]
    path = args.run / "checkpoints" / f"step-{chosen:04d}"
    parts, baseline, _ = model.load_models(model.validate_install(args.project_root))
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
    parser.add_argument("command", choices=("audit-cpu", "fit", "test"))
    for name in ("project-root", "dataset", "data-audit", "cpu-receipt"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    ensure_paths(args)
    if args.command == "audit-cpu":
        audit_cpu(args)
    else:
        if args.run is None:
            parser.error("GPU commands require --run")
        identity = admit_gpu(args)
        if args.command == "fit" and not args.resume:
            args.run.mkdir()
        with (args.run / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == "test":
                test_selected(args, identity)
            else:
                try:
                    raise SystemExit(run_fit(args, identity))
                except InterruptedError:
                    raise SystemExit(75) from None


if __name__ == "__main__":
    main()
