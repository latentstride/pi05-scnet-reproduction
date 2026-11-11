"""Build two review-only releases from the exact deployed unified archive."""

import argparse
import gzip
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from g2_author_h50_campaign import REVIEW_FILE
from g2_final_deploy import digest
from g2_standalone_deploy import (
    BASE_ARCHIVE,
    BASE_MANIFEST,
    MODES,
    SPEC_FILE,
    spec,
    verify_standalone,
)

CHANGED = ("scripts/g2_graphics.sh", "scripts/pace_g1_model_env.sh")
ADDED = ("g2_standalone_deploy.py", "g2_standalone_video.py")
DEPLOYED_ARX_ARCHIVE = "608982ae51ac552e93eb72635a96b53fa685a4f574af11b656b3e3b7b7d89043"
PROJECTION = "bounded_task_tcp_projection_v1"
PROJECTION_SCRIPTS = (
    "scripts/g2_remaining_conditions.py",
    "scripts/g2_remaining_coordinator.py",
    "scripts/g2_remaining_episode.py",
    "scripts/g2_task_runtime.py",
    "scripts/g2_task_tcp_projection.py",
)


def substitute(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"cannot uniquely identify frozen launcher setting: {old}")
    return text.replace(old, new)


def scope_caches(release):
    def scoped(file, anchor, setup, values):
        path = release / file
        text = substitute(path.read_text(), anchor, setup + anchor)
        for old, new in values:
            text = substitute(text, old, new)
        path.write_text(text)

    scoped(
        CHANGED[0],
        "make_sim_command() {",
        'cache="${G2_STANDALONE_CACHE:?}"\n',
        [
            (f"{key}={old}", f'"{key}=$cache/{new}"')
            for key, old, new in (
                ("XDG_CACHE_HOME", "/opt/robotwin/cache/xdg", "sim-xdg"),
                (
                    "TORCH_EXTENSIONS_DIR",
                    "/opt/robotwin/cache/torch_extensions",
                    "sim-torch",
                ),
                ("CUDA_CACHE_PATH", "/opt/robotwin/cache/cuda", "sim-cuda"),
                ("TMPDIR", "/opt/robotwin/tmp", "sim-tmp"),
                ("PYTHONPYCACHEPREFIX", "/opt/robotwin/cache/pycache", "sim-py"),
            )
        ],
    )
    scoped(
        CHANGED[1],
        "options=(--writable",
        'cache="${G2_STANDALONE_CACHE:?}"\n',
        [
            (f'"{key}=$root/cache/openpi-pace/{old}"', f'"{key}=$cache/{new}"')
            for key, old, new in (
                ("XDG_CACHE_HOME", "xdg", "model-xdg"),
                ("HF_HOME", "hf", "model-hf"),
                ("CUDA_CACHE_PATH", "cuda", "model-cuda"),
            )
        ],
    )


def write_archive(release, output, mode):
    manifest = release / "SHA256SUMS"
    files = sorted(p for p in release.rglob("*") if p.is_file() and p != manifest)
    manifest.write_text(
        "".join(f"{digest(p)}  {p.relative_to(release)}\n" for p in files)
    )
    verify_standalone(release)
    with (
        output.open("xb") as target,
        gzip.GzipFile(fileobj=target, filename="", mode="wb", mtime=0) as stream,
    ):
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for path in [manifest, *files]:
                info = archive.gettarinfo(
                    path,
                    str(Path(f"g2-standalone-{mode}") / path.relative_to(release)),
                )
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o755 if info.mode & 0o111 else 0o644
                with path.open("rb") as data:
                    archive.addfile(info, data)


def package_projection(project, parent, output):
    if digest(parent) != DEPLOYED_ARX_ARCHIVE:
        raise ValueError("expected the exact standalone archive for job 127510363")
    with TemporaryDirectory() as temporary:
        with tarfile.open(parent) as archive:
            archive.extractall(temporary, filter="data")
        release = Path(temporary).resolve() / "g2-standalone-arx"
        plan, selected = verify_standalone(release)
        episode = selected["episode"]
        config_path = f"configs/{episode}.json"
        config_file = release / config_path
        config = json.loads(config_file.read_text())
        config["task_tcp_projection"] = PROJECTION
        config_file.write_text(json.dumps(config, indent=2) + "\n")
        plan["trial_bindings"][episode]["config_sha256"] = digest(config_file)
        (release / "unified-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        # This declaration binds the old unified plan's exact bytes and is
        # consumed by the author-mode dispatch only.
        (release / REVIEW_FILE).unlink()
        base_path = release / "standalone-base-members.json"
        base = json.loads(base_path.read_text())
        for name in (*PROJECTION_SCRIPTS, config_path, "unified-plan.json", REVIEW_FILE):
            base["members"].pop(name, None)
        base_path.write_text(json.dumps(base, indent=2) + "\n")
        for name in PROJECTION_SCRIPTS:
            (release / name).write_bytes((project / name).read_bytes())
        write_archive(release, output, "arx")
        return dict(
            mode="arx",
            projection=PROJECTION,
            archive_sha256=digest(output),
            manifest_sha256=digest(release / "SHA256SUMS"),
            configuration_sha256=digest(release / SPEC_FILE),
            trial_config_sha256=digest(config_file),
            parent_archive_sha256=DEPLOYED_ARX_ARCHIVE,
            spec=selected,
        )


def package(project, parent, output, mode):
    if digest(parent) != BASE_ARCHIVE:
        raise ValueError("expected the exact 93b6c06 unified archive")
    selected = spec(mode)
    with TemporaryDirectory() as temporary:
        with tarfile.open(parent) as archive:
            archive.extractall(temporary, filter="data")
        release = Path(temporary).resolve() / "g2-author-h50-review"
        manifest = release / "SHA256SUMS"
        if digest(manifest) != BASE_MANIFEST:
            raise ValueError("frozen manifest mismatch")
        members = {}
        for line in manifest.read_text().splitlines():
            checksum, name = line.split("  ", 1)
            if digest(release / name) != checksum:
                raise ValueError(f"parent checksum mismatch: {name}")
            if name not in CHANGED:
                members[name] = checksum
        scope_caches(release)
        for name in ADDED:
            (release / "scripts" / name).write_bytes(
                (project / "scripts" / name).read_bytes()
            )
        (release / SPEC_FILE).write_text(json.dumps(selected, indent=2) + "\n")
        (release / "standalone-base-members.json").write_text(
            json.dumps(dict(manifest_sha256=BASE_MANIFEST, members=members), indent=2)
            + "\n"
        )
        launch = (release / "scripts/pace_g2_unified.slurm").read_text()
        launch = (
            launch.split("exec python3", 1)[0]
            + """exec python3 "$release/scripts/g2_standalone_deploy.py" "$release" \\
  --root "$root" --authorization "${G2_FINAL_APPROVAL:?allocation-bound approval file required}"
"""
        )
        minutes = selected["wall_s"] // 60
        (release / "scripts/pace_g2_standalone.slurm").write_text(
            launch.replace("07:00:00", f"{minutes // 60:02}:{minutes % 60:02}:00")
        )
        write_archive(release, output, mode)
        return dict(
            mode=mode,
            archive_sha256=digest(output),
            manifest_sha256=digest(manifest),
            configuration_sha256=digest(release / SPEC_FILE),
            frozen_arx_config_sha256=members.get(f"configs/{selected['episode']}.json"),
            spec=selected,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=MODES)
    p.add_argument("project", type=Path)
    p.add_argument("parent", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--arx-projection", action="store_true")
    a = p.parse_args()
    if a.arx_projection and a.mode != "arx":
        p.error("--arx-projection requires arx mode")
    receipt = (package_projection(a.project, a.parent, a.output) if a.arx_projection
               else package(a.project, a.parent, a.output, a.mode))
    print(json.dumps(receipt, indent=2))
