"""Read-only identity check for the preflight and compute-node probe."""

from g2_final_deploy import read
from g2_parallel_spec import digest, specs
from subprocess import check_output
from xml.etree import ElementTree

from g2_parallel_spec import CACHE, CALIBRATION, PACE


def query(*args):
    return check_output(args, text=True, timeout=25).strip()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def runtime_preflight(root, release):
    from pace_download import load_manifest, local_path

    r = read(release / "launch-spec.json")
    for folder in (
        "containers/openpi-ubuntu",
        "research/robotwin/robotwin-cu122/opt/robotwin/logs/BUILD_COMPLETE",
        "tools/pace-model-venv/bin/python",
    ):
        require((root / folder).exists(), f"missing deployed runtime: {folder}")
    for name, commit in (
        ("PACE", specs()[0]["pace_revision"]),
        ("Openpi-PACE", specs()[0]["openpi_pace_revision"]),
    ):
        source = root / "source" / f"{name}-{commit}"
        require(source.is_dir(), f"{name} missing")
        git = ("git", f"--git-dir={source / '.git'}", f"--work-tree={source}")
        require(
            query(*git, "rev-parse", "HEAD") == commit
            and not query(*git, "status", "--porcelain", "-uno"),
            f"{name} tracked source or revision differs",
        )
    pace = root / "source" / f"PACE-{PACE}"
    for name, expected in (
        ("pace/configs/expert_scenes.json", CACHE),
        ("pace/configs/calibration/handover_block.json", CALIBRATION),
    ):
        require(digest(pace / name) == expected, f"author asset differs: {name}")
    audit = read(release / "asset-audit.json")
    robot = next(row for row in audit["robots"] if row["robot"] == "aloha-agilex")
    assets = pace / "assets"
    urdf = assets / robot["urdf_member"]
    require(digest(urdf) == robot["urdf_sha256"], "Aloha URDF differs")
    meshes = {
        urdf.parent / m.attrib["filename"] for m in ElementTree.parse(urdf).iter("mesh")
    }
    require(
        meshes
        and all(p.is_file() for p in meshes)
        and all(
            any((assets / name).iterdir()) for name in ("objects", "background_texture")
        ),
        "required Aloha meshes or scene assets are missing",
    )
    model = load_manifest()
    require(
        model == read(release / "pace_download_manifest.json"), "model manifest differs"
    )
    receipt_path = root / r["model_cpu_receipt"].split("<PROJECT_ROOT>/", 1)[1]
    require(
        digest(receipt_path) == r["model_cpu_receipt_sha256"],
        "frozen model CPU receipt hash differs",
    )
    receipt = read(receipt_path)
    cache = root / "cache/pace-handover"
    snapshot = {}
    for item in model["files"]:
        path = local_path(cache, item["path"])
        stat = path.stat()
        snapshot[item["path"]] = [stat.st_size, stat.st_mtime_ns, item["sha256"]]
    require(
        snapshot == receipt["checkpoint_files"]
        and digest(cache / model["norm_stats"]) == model["norm_stats_sha256"]
        and digest(root / "cache/openpi-pace/big_vision/paligemma_tokenizer.model")
        == snapshot["tokenizer.model"][2],
        "checkpoint, norm stats or tokenizer differs",
    )
