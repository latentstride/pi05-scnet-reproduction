"""Build six deterministic standalone candidate archives from the frozen author release."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory

from g2_standalone_matrix import CONFIG, selection, trial_budget, verify_package
from g2_parallel_spec import BATCH_SCHEME, digest, extract, load, specs
from g2_standalone_deploy import (
    CAMPAIGN_RESERVE_S,
    allocation_minimum,
    execution_overhead,
    verify_standalone,
)
from g2_standalone_submit import seconds

PARENT_ARCHIVE = "1d60de6eeaffacab0962119186d8bc7aac4755537cfba1aa900c325664585a6e"
BASE = "a5a67cfb27a787da5562f82e03c3e8d67c257ec3"
SOURCE_BATCH = "616fc754e36acc5b2c89b6fb302eb254b340b751e4834fe84368d1ad74fb87e3"
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


def cpu_receipt(project, key, conf, shell, selected):
    research = project / "research/g2"
    source_batch = research / "standalone-seven/batch.json"
    if digest(source_batch) != SOURCE_BATCH:
        raise ValueError("historical CPU provenance manifest differs")
    (source,) = [
        row for row in json.loads(source_batch.read_text())["rows"] if row["id"] == key
    ]
    old_conf = research / "parallel-batch/configs" / f"{key}.json"
    old_receipt = research / "parallel-batch/cpu" / f"{key}.json"
    prior = json.loads(old_receipt.read_text())
    old = json.loads(old_conf.read_text())
    spec = load(conf)
    allowed = {"wall_s", "trial_soft_s", "trial_hard_s"}
    if (
        digest(old_conf) != source["config_sha256"]
        or digest(old_receipt) != source["cpu_sha256"]
        or prior.get("admission") != "PASS"
        or prior.get("config_sha256") != source["config_sha256"]
        or old.keys() != spec.keys()
        or dict(old, resources=spec["resources"]) != spec
        or old["resources"].keys() != spec["resources"].keys()
        or {k for k, v in old["resources"].items() if v != spec["resources"][k]}
        != allowed
    ):
        raise ValueError(f"{key} previous CPU evidence is inapplicable")
    budget = trial_budget(spec)
    overhead = execution_overhead(selected)
    minimum = allocation_minimum(selected, overhead)
    directives = dict(
        line.removeprefix("#SBATCH ").split("=", 1)
        for line in shell.read_text().splitlines()
        if line.startswith("#SBATCH ") and "=" in line
    )
    if (
        seconds(directives["--time"]) != selected["wall_s"]
        or seconds(directives["--time-min"]) != selected["wall_s"]
        or directives["--signal"] != "B:TERM@120"
        or minimum > selected["wall_s"]
        or overhead["statistics_s"] + overhead["cleanup_s"]
        != spec["resources"]["cleanup_s"]
        or CAMPAIGN_RESERVE_S < 120
    ):
        raise ValueError(f"{key} inconsistent standalone resource envelope")
    return dict(
        id=key,
        admission="PASS",
        config_sha256=digest(conf),
        evidence_scope="prior CPU scene and policy substitute; fresh resource ledger check",
        source_batch_sha256=SOURCE_BATCH,
        source_config_sha256=source["config_sha256"],
        source_cpu_sha256=source["cpu_sha256"],
        resource_acceptance=dict(
            overhead=overhead,
            host_reserve_s=CAMPAIGN_RESERVE_S,
            slurm_signal_s=120,
            minimum_s=minimum,
            unassigned_s=selected["wall_s"] - minimum,
            soft_s=budget.soft_s,
            hard_s=budget.hard_s,
            request_s=budget.request_s,
            physics_steps=budget.physics_steps,
        ),
    )


def package(project, parent, evidence, output):
    if digest(parent) != PARENT_ARCHIVE:
        raise ValueError("expected the exact prior standalone author archive")
    output.mkdir(parents=True, exist_ok=True)
    batch = dict(
        baseline=BASE,
        submission_allowed=False,
        submission_scheme=BATCH_SCHEME,
        rows=[],
    )
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
            # Bind historical CPU observations to this candidate via a fresh budget check.
            receipt = release / "parallel-cpu-receipt.json"
            selected = selection(spec)
            receipt.write_text(
                json.dumps(
                    cpu_receipt(
                        project,
                        key,
                        conf,
                        release / "scripts/pace_g2_parallel.slurm",
                        selected,
                    ),
                    indent=2,
                )
                + "\n"
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
                    cpu_sha256=digest(receipt),
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
