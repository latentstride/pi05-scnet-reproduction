"""Build seven deterministic standalone candidate archives from the frozen author release."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory

from g2_standalone_matrix import CONFIG, verify_package
from g2_parallel_spec import digest, extract, specs
from g2_standalone_deploy import verify_standalone

PARENT_ARCHIVE = "1d60de6eeaffacab0962119186d8bc7aac4755537cfba1aa900c325664585a6e"
BASE = "a5a67cfb27a787da5562f82e03c3e8d67c257ec3"
OVERLAY = (
    "g2_parallel_spec.py",
    "g2_standalone_deploy.py",
    "g2_standalone_matrix.py",
    "g2_final_campaign.py",
    "g2_standalone_startup.py",
    "g2_parallel_bootstrap.py",
    "g2_parallel_worker.py",
    "g2_author_native_entry.py",
    "g2_author_native_h50.py",
    "g2_author_physx_observer.py",
    "pace_g2_parallel.slurm",
)


def package(project, parent, evidence, output):
    if digest(parent) != PARENT_ARCHIVE:
        raise ValueError("expected the exact prior standalone author archive")
    output.mkdir(parents=True, exist_ok=True)
    batch = dict(baseline=BASE, submission_allowed=False, rows=[])
    with TemporaryDirectory() as tmp:
        parent_dir = Path(tmp).resolve() / "g2-standalone-author"
        extract(parent, Path(tmp), "g2-standalone-author")
        verify_standalone(parent_dir)
        for spec in specs():
            key = spec["id"]
            name = f"g2-parallel-{key}"
            release = Path(tmp).resolve() / name
            shutil.copytree(parent_dir, release)
            for filename in OVERLAY:
                shutil.copyfile(
                    project / "scripts" / filename, release / "scripts" / filename
                )
            for old in (
                "standalone-base-members.json",
                "standalone-review.json",
                "scripts/pace_g2_standalone.slurm",
                "scripts/pace_g2_unified.slurm",
            ):
                (release / old).unlink()
            graphics = release / "scripts/g2_graphics.sh"
            old_root = 'cd "${SLURM_SUBMIT_DIR:?Submit from personal project root}"'
            new_root = (
                'cd "${G2_PROJECT_ROOT:?Explicit parallel project root required}"'
            )
            script = graphics.read_text()
            if script.count(old_root) != 1:
                raise ValueError("expected one graphics working-directory binding")
            graphics.write_text(script.replace(old_root, new_root))
            conf = evidence / "configs" / f"{key}.json"
            shutil.copyfile(conf, release / CONFIG)
            (release / "parallel-admission.json").write_text(
                json.dumps(
                    dict(submission_allowed=False, automatic_retry=False, spec=spec),
                    indent=2,
                )
                + "\n"
            )
            receipt = evidence / "cpu" / f"{key}.json"
            if receipt.is_file():
                shutil.copyfile(receipt, release / "parallel-cpu-receipt.json")
            if key == "G7":
                shutil.copyfile(
                    evidence / "cpu/G7-reference.json",
                    release / "parallel-reset-reference.json",
                )
            manifest = release / "SHA256SUMS"
            members = sorted(
                p for p in release.rglob("*") if p.is_file() and p != manifest
            )
            manifest.write_text(
                "".join(f"{digest(p)}  {p.relative_to(release)}\n" for p in members)
            )
            verify_package(release)
            archive_path = output / f"{name}.tar.gz"
            with (
                archive_path.open("xb") as target,
                gzip.GzipFile(
                    fileobj=target, filename="", mode="wb", mtime=0
                ) as stream,
                tarfile.open(fileobj=stream, mode="w") as archive,
            ):
                for p in [manifest, *members]:
                    member = archive.gettarinfo(
                        p, str(Path(name) / p.relative_to(release))
                    )
                    member.uid = member.gid = member.mtime = 0
                    member.uname = member.gname = ""
                    member.mode = 0o644
                    with p.open("rb") as data:
                        archive.addfile(member, data)
            batch["rows"].append(
                dict(
                    id=key,
                    name=name,
                    config_sha256=digest(conf),
                    archive_sha256=digest(archive_path),
                    manifest_sha256=digest(manifest),
                    cpu_sha256=digest(receipt) if receipt.is_file() else None,
                    members=len(members),
                )
            )
    (output / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    return batch


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("project", "parent", "evidence", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    package(a.project, a.parent, a.evidence, a.output)
