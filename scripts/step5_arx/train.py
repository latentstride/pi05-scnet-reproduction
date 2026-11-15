"""CPU checkpoint audit, gated single-GPU fit, and paired held-out evaluation."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import sys
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from . import admission, model
from .adapter import MODEL_CAMERAS
from .config import CODE_ROOT, MANIFEST, OPENPI, PACE_MODEL, SETTINGS, write_json
from .data import ARM, GRIP, Dataset, digest, partitions

MIN_DISK_BYTES = 16 * 2**30


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
    for name in ("dataset", "data_audit", "cpu_receipt", "run", "stop_file", "job_record", "evidence"):
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


def lower_initial_loss(loss, initial, frozen, rng, observation, target):
    # JAX's AOT metadata reconstructs argument trees with ArgInfo leaves.
    # Pass flat arrays at that boundary and rebuild the typed Observation in tracing.
    leaves, structure = jax.tree.flatten(observation)

    @jax.jit
    def invoke(state, key, obs_leaves, actions):
        return loss(state, key, jax.tree.unflatten(structure, obs_leaves), actions)

    return invoke.lower((initial, frozen), rng, leaves, target), leaves


def check_initial_loss(loss, vanilla_loss, initial, frozen, base_state, rng, observation, target, path):
    """Gate zero residual and return an evaluator for the same batch and executable."""
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
    np.testing.assert_allclose(seeded, zeroed, rtol=rtol, atol=atol)
    return seeded, evaluate_adapters


def update(step, train, frozen, optimizer_state, key, observation, target):
    fitted, optimizer_state, values = step(train, frozen, optimizer_state, key, observation, target)
    loss, grad, change = map(float, values)
    if not np.isfinite([loss, grad, change]).all() or grad <= 0 or change <= 0:
        raise RuntimeError(f"Invalid optimizer update: loss={loss}, grad={grad}, change={change}")
    return fitted, optimizer_state, dict(loss=loss, grad_norm=grad, adapter_delta_norm=change)


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
            ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "optax",
             "orbax-checkpoint", "h5py", "numpy", "pillow")}


def runtime_identity():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                libc=os.confstr("CS_GNU_LIBC_VERSION"),
                container_os_sha256=digest(Path("/etc/os-release")),
                apptainer=os.environ.get("STEP5_APPTAINER_VERSION", ""))


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
        raise RuntimeError("CPU receipt requires a CPU-only JAX device")
    if args.cpu_receipt.exists():
        raise FileExistsError(args.cpu_receipt)
    snapshot = model.checkpoint_snapshot(args.project_root)
    params = model.validate_install(args.project_root, full_hash=True)
    identity = fingerprint(args)
    print("CPU manifest and dataset verified", flush=True)
    graph, train, frozen, optimizer, opt, baseline, report = model.cpu_checkpoint_check(params, platform())
    jax.block_until_ready((train, opt))
    print("CPU graph, 5 Orbax tensor reads and optimizer initialized", flush=True)
    with Dataset(args.dataset) as dataset:
        policy = model.NativePolicy(dataset.norm_stats)
        indices = tuple(dataset.indices("train"))
        episode, row = indices[len(indices) // 2]
        sample = dataset.read("train", episode, row)
        obs, target = policy.batch([sample])
        key = jax.random.key(SETTINGS.seed)
        input_report = arx_input_contract(obs, target, key)
        input_report.update(episode=episode, row=row)
        write_json(args.cpu_receipt.with_name("arx-input-contract.json"), input_report)
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
    with tempfile.TemporaryDirectory(dir=args.cpu_receipt.parent) as temp:
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
    write_json(args.cpu_receipt, dict(status=admission.CPU_STRUCTURAL_ONLY, identity=identity, graph=report, versions=versions(),
                                     checkpoint_snapshot=snapshot, checkpoint_recovery_checked=True,
                                     model_input_shape=list(target.shape), abstract_update_traced=True,
                                     token_count=input_report["valid_prompt_tokens"], input_contract=input_report,
                                     full_initial_eval_lowered=True, full_eval_and_sampling_traced=True))


def validate_admission_evidence(args, packet, identity):
    status = packet["status"]
    structural = admission.verify_evidence(args.project_root, packet["evidence"], status)
    structural_identity = structural["identity"]
    if status == admission.CPU_NUMERIC_EXCEPTION:
        # Reuse recorded structure only when the upstream model, data and settings match.
        # The exception packet itself binds the new runtime code fingerprint exactly.
        structural_identity = {**structural_identity, "code_sha256": identity["code_sha256"]}
    if (structural_identity != identity or structural["versions"] != packet["versions"]
            or structural["checkpoint_snapshot"] != packet["checkpoint_snapshot"]):
        raise ValueError("Structural evidence does not match the admitted model, data or environment")
    if status == admission.CPU_VERIFIED:
        record = json.loads(Path(packet["evidence"]["cpu_verified_numeric"]["path"]).read_text())
        steps = record.get("steps", [])
        if (record.get("identity") != identity or record.get("status") != "passed"
                or record.get("resumed_matches_uninterrupted") is not True
                or record.get("frozen_unchanged") is not True
                or record.get("model_input_shape") != [1, 50, 32] or len(steps) != 2):
            raise ValueError("Missing complete CPU numerical verification")
        for i, row in enumerate(steps, 1):
            values = [row[k] for k in ("loss", "grad_norm", "adapter_delta_norm")]
            if not np.isfinite(values).all() or min(values[1:]) <= 0 or row.get("adamw_count") != i:
                raise ValueError("Invalid CPU numerical update evidence")


def inspect_admission(args):
    if not args.run.parent.is_dir() or shutil.disk_usage(args.run.parent).free < MIN_DISK_BYTES:
        raise RuntimeError("Need an existing run parent with 16 GiB free")
    receipt = admission.canonical_under(args.project_root, args.cpu_receipt)
    raw = receipt.read_bytes()
    packet = json.loads(raw)
    status = packet.get("status")
    if status not in {admission.CPU_VERIFIED, admission.CPU_NUMERIC_EXCEPTION}:
        raise PermissionError("A complete CPU verification or approved numerical exception is required")
    if status == admission.CPU_NUMERIC_EXCEPTION:
        if args.command not in {"fit", "preflight"}:
            raise PermissionError("The numerical exception is limited to one fit job")
        admission.verify_approval(args.project_root, packet, receipt, raw)
        if packet.get("unresolved") != "CPU_NUMERIC_UNRESOLVED":
            raise ValueError("CPU numerical exception must identify the unresolved result")
        scope = packet["scope"]
        admission.verify_scope(args, scope)
        if admission.claim_path(args.project_root, scope["token"]).exists():
            raise PermissionError("CPU numerical exception has already been consumed")
        if args.command == "preflight" and admission.reservation_path(args.project_root, scope["token"]).exists():
            raise PermissionError("CPU numerical exception has already reserved its sole submission")
    identity = fingerprint(args)
    if (packet.get("identity") != identity or packet.get("versions") != versions()
            or packet.get("runtime") != runtime_identity()
            or packet.get("release") != admission.release_identity(args.project_root, CODE_ROOT)):
        raise RuntimeError("Admission receipt is missing or stale for this release, dataset or runtime")
    model.validate_install(args.project_root)
    if packet["checkpoint_snapshot"] != model.checkpoint_snapshot(args.project_root):
        raise RuntimeError("Model file snapshot differs from the CPU-verified files")
    validate_admission_evidence(args, packet, identity)
    if args.command in {"fit", "preflight"}:
        if args.resume:
            if json.loads((args.run / "identity.json").read_text()) != identity:
                raise RuntimeError("Cannot resume with a different dataset, code or config")
        elif args.run.exists():
            raise FileExistsError(args.run)
    return identity, packet, hashlib.sha256(raw).hexdigest()


def prepare_exception(args):
    if platform().platform != "cpu":
        raise RuntimeError("Prepare the review in the CPU container")
    if args.cpu_receipt.exists() or (args.run.exists() and not args.resume):
        raise FileExistsError("The review requires a new packet and an unused run unless resuming")
    if not args.run.parent.is_dir() or shutil.disk_usage(args.run.parent).free < MIN_DISK_BYTES:
        raise RuntimeError("Need an existing run parent with 16 GiB free")
    model.validate_install(args.project_root)
    identity = fingerprint(args)
    if args.resume and json.loads((args.run / "identity.json").read_text()) != identity:
        raise RuntimeError("Cannot resume with a different dataset, code or config")
    packet = dict(status=admission.CPU_NUMERIC_EXCEPTION, unresolved="CPU_NUMERIC_UNRESOLVED",
                  authorization=dict(state=admission.PENDING), identity=identity, versions=versions(),
                  runtime=runtime_identity(),
                  release=admission.release_identity(args.project_root, CODE_ROOT),
                  checkpoint_snapshot=model.checkpoint_snapshot(args.project_root),
                  scope=admission.make_scope(args, secrets.token_hex(16), args.job_account, args.slurm_log),
                  evidence=json.loads(args.evidence.read_text()))
    validate_admission_evidence(args, packet, identity)
    write_json(args.cpu_receipt, packet)
    print(json.dumps(dict(status="PENDING_USER_APPROVAL", release=packet["release"],
                          receipt=str(args.cpu_receipt), code_sha256=identity["code_sha256"])), flush=True)
    # This is a review artifact. Only preflight on a signed approval returns the runnable command.
    print(admission.submission_command(args, CODE_ROOT, packet["scope"]), flush=True)


def admit_gpu(args):
    identity, approved, receipt_sha = inspect_admission(args)
    exception = approved["status"] == admission.CPU_NUMERIC_EXCEPTION
    if exception:
        job_id = os.environ.get("STEP5_SLURM_JOB_ID")
        if args.job_record is None:
            raise PermissionError("The CPU exception requires the actual Slurm allocation record")
        admission.verify_reservation(args.project_root, approved["scope"], receipt_sha)
        admission.validate_slurm_record(args.job_record.read_text(), job_id, approved["scope"], CODE_ROOT)
    cuda = model.cuda_identity()
    print(json.dumps(dict(stage="cuda_device", **cuda)), flush=True)
    device = platform()
    if device.platform != "gpu" or "RTX 4090" not in device.device_kind.upper():
        raise RuntimeError(f"Expected one allocated RTX 4090, found {device.device_kind}")
    print(json.dumps(dict(gpu_device=str(device), gpu_model=device.device_kind,
                          memory=device.memory_stats())), flush=True)
    if exception:
        admission.consume(args.project_root, approved["scope"], job_id, receipt_sha)
        print(json.dumps(dict(stage="cpu_numerical_exception_consumed", job_id=job_id,
                              token=approved["scope"]["token"], receipt_sha256=receipt_sha)), flush=True)
    return identity


def stop_requested(args):
    return args.stop_file is not None and args.stop_file.stat().st_size > 0


def record_gpu_progress(run, stage, **details):
    event = dict(stage=stage, **details)
    write_json(run / "gpu_progress.json", event)
    print(json.dumps(event), flush=True)


def frozen_digests(frozen):
    # Transfer one tensor at a time. This allocates no second frozen tree on CUDA.
    leaves, _ = jax.tree_util.tree_flatten_with_path(frozen)
    digests = {}
    for path, value in leaves:
        host = np.ascontiguousarray(jax.device_get(value))
        digests[jax.tree_util.keystr(path)] = hashlib.sha256(memoryview(host.view(np.uint8))).hexdigest()
        del host
    return digests


def gate_update(run, stage, index, step, train, frozen, opt, key, obs, target, gpu):
    record_gpu_progress(run, stage, step=index, status="running")
    fitted, opt, metrics = update(step, train, frozen, opt, key, obs, target)
    count = int(opt[1][0].count)
    if count != index:
        raise RuntimeError(f"{stage} AdamW count {count} does not match step {index}")
    row = dict(step=index, adamw_count=count, memory=gpu.memory_stats(), **metrics)
    write_json(run / f"{stage}_{index:04d}.json", row)
    record_gpu_progress(run, stage, status="passed", **row)
    return fitted, opt, row


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
    gpu = platform()
    record_gpu_progress(run, "gpu_checkpoint_restore", status="running")
    params = model.validate_install(args.project_root)
    parts, baseline, report = model.load_models(params)
    graph, initial, frozen, optimizer, initial_opt = parts
    base_graph, base_state = baseline
    step = optimizer_step(graph, optimizer)
    loss = flow_loss(graph, adapted=True)
    pred = model.sampler(graph, adapted=True)
    rng = jax.random.key(SETTINGS.seed)
    write_json(run / "parameter_layout.json", report)
    record_gpu_progress(run, "gpu_checkpoint_restore", status="passed", memory=gpu.memory_stats())
    with Dataset(args.dataset) as dataset:
        policy = model.NativePolicy(dataset.norm_stats)
        indices = tuple(dataset.indices("train"))
        overfit = [dataset.read("train", *indices[len(indices) // 2])]
        obs, target = policy.batch(overfit)
        smoke = run / "gpu_gates.json"
        if not smoke.exists():
            record_gpu_progress(run, "gpu_initial_loss", status="running")
            first_val, fixed_eval = check_initial_loss(
                loss, flow_loss(base_graph, adapted=False), initial, frozen, base_state, rng, obs, target,
                run / "initial_loss.json")
            if stopped():
                return 75
            record_gpu_progress(run, "gpu_frozen_check_before", status="running")
            initial_frozen = frozen_digests(frozen)
            fitted, opt = initial, initial_opt
            validation_keys = (rng, jax.random.fold_in(rng, 2))
            stage_a = []
            for i, key in enumerate(validation_keys, 1):
                if stopped():
                    return 75
                fitted, opt, row = gate_update(run, "gpu_stage_a", i, step, fitted, frozen, opt,
                                               key, obs, target, gpu)
                stage_a.append(row)
            record_gpu_progress(run, "gpu_frozen_check_after", status="running")
            if frozen_digests(frozen) != initial_frozen:
                raise RuntimeError("Frozen base changed during GPU validation")
            write_json(run / "gpu_frozen_unchanged.json", dict(leaves=len(initial_frozen), sha256=initial_frozen))
            if stopped():
                return 75
            # Both experimental gates leave the formal initial optimizer state untouched.
            fitted, opt = initial, initial_opt
            overfit_log = []
            for i in range(SETTINGS.overfit_steps):
                fitted, opt, row = gate_update(run, "gpu_overfit", i + 1, step, fitted, frozen, opt,
                                               jax.random.fold_in(rng, i), obs, target, gpu)
                overfit_log.append(row)
                if stopped():
                    return 75
            record_gpu_progress(run, "gpu_overfit_eval", status="running")
            last_val = fixed_eval(fitted)
            del fixed_eval
            relative = (first_val - last_val) / max(first_val, 1e-12)
            gates = dict(stage_a=stage_a, frozen_unchanged=True, overfit=overfit_log, initial_eval_loss=first_val,
                         final_eval_loss=last_val, relative_improvement=relative, memory=gpu.memory_stats())
            if not np.isfinite(last_val) or relative < SETTINGS.overfit_min_relative_improvement:
                if not np.isfinite(last_val):
                    gates.update(final_eval_loss=str(last_val), relative_improvement=str(relative))
                write_json(run / "gpu_gates_failed.json", gates)
                raise RuntimeError(f"20-step overfit gate failed: relative reduction {relative:.4g}")
            write_json(smoke, dict(status="passed", **gates))
            record_gpu_progress(run, "gpu_gates_passed", memory=gpu.memory_stats())
            del fitted, opt
        else:
            receipt = json.loads(smoke.read_text())
            if (receipt.get("status") != "passed" or receipt.get("frozen_unchanged") is not True
                    or [step["adamw_count"] for step in receipt.get("stage_a", [])] != [1, 2]
                    or len(receipt.get("overfit", [])) != SETTINGS.overfit_steps):
                raise RuntimeError("GPU gate receipt is invalid")
        if stopped():
            return 75
        val_baseline = run / "val-frozen.json"
        if not val_baseline.exists():
            record_gpu_progress(run, "gpu_frozen_validation", status="running")
            write_json(val_baseline, evaluate(dataset, "val", policy, model.sampler(base_graph, adapted=False),
                                             flow_loss(base_graph, adapted=False), base_state, stopped=stopped))
        last = latest_checkpoint(dir_ckpt)
        start = int(last.name.split("-")[-1]) if last else 0
        fitted, opt = restore(last, initial, initial_opt) if last else (initial, initial_opt)
        for i in range(start, SETTINGS.train_steps + 1):
            if i > start:
                record_gpu_progress(run, "gpu_formal_train", step=i, status="running")
                index = np.random.default_rng(np.random.SeedSequence([SETTINGS.seed, i])).integers(len(indices))
                obs, target = policy.batch([dataset.read("train", *indices[index])])
                fitted, opt, metrics = update(step, fitted, frozen, opt, jax.random.fold_in(rng, i), obs, target)
                count = int(opt[1][0].count)
                if count != i:
                    raise RuntimeError(f"Formal AdamW count {count} does not match step {i}")
                print(json.dumps(dict(step=i, adamw_count=count, **metrics)), flush=True)
            if i and (i % SETTINGS.checkpoint_interval == 0 or stopped() or i == SETTINGS.train_steps):
                checkpoint = dir_ckpt / f"step-{i:04d}"
                if not complete_checkpoint(checkpoint):
                    record_gpu_progress(run, "gpu_checkpoint_save", step=i, status="running")
                    save(checkpoint, fitted, opt)
            if stopped():
                return 75
            if i and (i % SETTINGS.validation_interval == 0 or i == SETTINGS.train_steps):
                out = run / f"val-{i:04d}.json"
                if not out.exists():
                    record_gpu_progress(run, "gpu_validation", step=i, status="running")
                    write_json(out, evaluate(dataset, "val", policy, pred, loss, (fitted, frozen), stopped=stopped))
        reports = [(int(path.stem.split("-")[-1]), json.loads(path.read_text())["episode_mean"])
                   for path in run.glob("val-[0-9][0-9][0-9][0-9].json")]
        best, metrics = min(reports, key=lambda row: (row[1]["flow_loss"], row[1]["arm_mae_rad"],
                                                      row[1]["gripper_mae"], row[0]))
        write_json(run / "selected.json", dict(step=best, metrics=metrics, last_step=SETTINGS.train_steps,
                                               memory=gpu.memory_stats()))
        record_gpu_progress(run, "gpu_fit_complete", step=SETTINGS.train_steps, memory=gpu.memory_stats())
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
    parser.add_argument("command", choices=("audit-cpu", "prepare-exception", "preflight", "fit", "test"))
    for name in ("project-root", "dataset", "data-audit", "cpu-receipt"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--job-record", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--job-account")
    parser.add_argument("--slurm-log", type=Path)
    parser.add_argument("--reserve-submission", action="store_true")
    args = parser.parse_args()
    ensure_paths(args)
    if args.reserve_submission and args.command != "preflight":
        parser.error("--reserve-submission is only accepted by preflight")
    if args.command == "audit-cpu":
        audit_cpu(args)
    elif args.command == "prepare-exception":
        if not all((args.run, args.evidence, args.job_account, args.slurm_log)):
            parser.error("prepare-exception requires --run, --evidence, --job-account and --slurm-log")
        prepare_exception(args)
    elif args.command == "preflight":
        if args.run is None:
            parser.error("preflight requires --run")
        _, approved, receipt_sha = inspect_admission(args)
        if approved["status"] == admission.CPU_NUMERIC_EXCEPTION:
            if args.reserve_submission:
                admission.reserve_submission(args.project_root, approved["scope"], receipt_sha)
            print(admission.submission_command(args, CODE_ROOT, approved["scope"]))
        elif args.reserve_submission:
            parser.error("Only an approved numerical exception uses a submission reservation")
    else:
        if args.run is None:
            parser.error("GPU commands require --run")
        print(json.dumps(dict(stage="gpu_preflight", status="running")), flush=True)
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
                except Exception as error:
                    write_gpu_failure(args.run, error)
                    raise


if __name__ == "__main__":
    main()
