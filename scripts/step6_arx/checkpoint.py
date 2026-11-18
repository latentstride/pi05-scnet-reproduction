"""Read-only checks of a selected Step 5 training result."""

from hashlib import sha256
import json


class CheckpointUnavailable(ValueError):
    pass


def file_sha(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def inspect_checkpoint(run, plan):
    selected_path = run / "selected.json"
    if not selected_path.is_file():
        raise CheckpointUnavailable(f"missing Step 5 selection: {selected_path}")
    selected = read_json(selected_path)
    identity = read_json(run / "identity.json")
    expected = {"dataset_sha256": plan["dataset_sha256"], "pace_checkpoint": plan["checkpoint_revision"]}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise CheckpointUnavailable("training identity differs from the Step 6 data or PACE revision")
    last_step = identity.get("settings", {}).get("train_steps")
    step = selected.get("step")
    if (type(last_step) is not int or selected.get("last_step") != last_step
            or type(step) is not int or not 0 < step <= last_step):
        raise CheckpointUnavailable("training selection is incomplete or invalid")
    progress_path = run / "gpu_progress.json"
    if not progress_path.is_file():
        raise CheckpointUnavailable(f"missing Step 5 completion record: {progress_path}")
    progress = read_json(progress_path)
    if progress.get("stage") != "gpu_fit_complete" or progress.get("step") != last_step:
        raise CheckpointUnavailable("Step 5 training did not complete the configured steps")
    checkpoint = run / "checkpoints" / f"step-{step:04d}"
    for candidate in (checkpoint, run / "checkpoints" / f"step-{last_step:04d}"):
        if not (candidate / "_METADATA").is_file():
            raise CheckpointUnavailable(f"missing Orbax metadata: {candidate}")
        committed = read_json(candidate / "_CHECKPOINT_METADATA").get("commit_timestamp_nsecs")
        if type(committed) is not int or committed <= 0:
            raise CheckpointUnavailable(f"incomplete Orbax checkpoint: {candidate}")
        if not (candidate / "manifest.ocdbt").is_file():
            raise CheckpointUnavailable(f"missing Orbax manifest: {candidate}")
    return selected, identity, checkpoint
