"""Frozen six-condition, four-hour design. No scheduler or model dependencies."""

from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from tempfile import TemporaryDirectory

CHECKPOINT = "17ef545270cf26ed81169893b60f845f80ce6557"
NORM = "8a1f114a8873dfa0788c7f512851ee3063224ef2b8af270421076a2e4e544960"
CACHE = "12ac49e3913308901113cc8ddd9f71f0987ce8c9ebd83f03c308c759bdbeb939"
CALIBRATION = "d66b55926fb3afc655199986710f84d8bc519e4443c11ceb51725814da98204f"
PACE = "fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
OPENPI = "f4e52e6de4cc487a11585a3a7414d610b2883dae"
TASK = "handover_block"
SCENES = (100000, 200000, 300000)
RESOURCES = dict(
    partition="ksagnormal01",
    node="gnode39",
    gpu="RTX4090-PCIE-24GB-LS",
    cpus=4,
    memory_mib=32768,
    gpus=1,
    wall_s=14400,
    graphics_s=300,
    model_start_s=900,
    trial_soft_s=11400,
    trial_hard_s=12000,
    cleanup_s=480,
    request_s=390,
)
BATCH_SIZE = 2 * len(SCENES)
BATCH_SCHEME = "standalone-six-4h-v1"
MAX_RATE = Decimal("3.60")
JOB_FEE_CAP = MAX_RATE * RESOURCES["wall_s"] / 3600
BATCH_FEE_CAP = JOB_FEE_CAP * BATCH_SIZE
SELECTION = "first expert-accepted scene of each fixed cache shard (index 0)"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def extract(archive_path, directory, expected_name):
    """Extract only ordinary, unique files from a verified package."""
    with (
        TemporaryDirectory(dir=directory) as temp,
        tarfile.open(archive_path) as archive,
    ):
        root = Path(temp)
        seen = set()
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or len(path.parts) < 2
                or path.parts[0] != expected_name
                or path.as_posix() != member.name
                or path in seen
            ):
                raise ValueError("unexpected archive member")
            seen.add(path)
            destination = root.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (
                archive.extractfile(member) as source,
                destination.open("xb") as target,
            ):
                shutil.copyfileobj(source, target)
        (root / expected_name).rename(directory / expected_name)


def specs():
    result = []
    for i, (mode, method, shard) in enumerate(
        [("author_native", "fixed_h50", j) for j in range(len(SCENES))]
        + [("author_native", "pace", j) for j in range(len(SCENES))],
        1,
    ):
        key = f"g{i}-{mode}-{method}-{SCENES[shard]}"
        result.append(
            dict(
                id=f"G{i}",
                episode_id=key,
                output_dir=f"results/pace-g2-parallel/{key}",
                mode=mode,
                method=method,
                scene_cache_shard=shard,
                scene_cache_index=0,
                scene_seed=SCENES[shard],
                scene_cache_sha256=CACHE,
                calibration_sha256=CALIBRATION,
                model_rng_seed=0,
                task=TASK,
                checkpoint_revision=CHECKPOINT,
                norm_stats_sha256=NORM,
                pace_revision=PACE,
                openpi_pace_revision=OPENPI,
                action_cap=800,
                window_cap=50,
                resources=RESOURCES,
                port_base=20000 + (i - 1) * 2000,
            )
        )
    return result


def load(path):
    item = json.loads(path.read_text())
    if item not in specs():
        raise ValueError("configuration differs from the frozen six-condition design")
    return item


def cache_row(scene, load_cached_scenes):
    shard, index = scene["scene_cache_shard"], scene["scene_cache_index"]
    rows, meta = load_cached_scenes(TASK, shard, index + 1)
    row = rows[index]
    if (
        meta["sha256"] != scene["scene_cache_sha256"]
        or row["scene_seed"] != scene["scene_seed"]
        or not isinstance(row["episode_info"], dict)
    ):
        raise ValueError("author cache identity differs")
    return row, meta


def legacy_scene():
    return specs()[0]


def write(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for spec in specs():
        (directory / f"{spec['id']}.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    import sys

    write(Path(sys.argv[1]))
