#!/usr/bin/env python3
"""Plan, verify or explicitly download the pinned PACE handover_block assets."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import quote

_RELEASE_MANIFEST = Path(__file__).resolve().with_name("pace_download_manifest.json")
MANIFEST = (_RELEASE_MANIFEST if _RELEASE_MANIFEST.is_file()
            else Path(__file__).resolve().parents[1] / "pace_download_manifest.json")


def load_manifest():
    manifest = json.loads(MANIFEST.read_text())
    if (manifest["schema_version"] != 1 or manifest["repo"] != "niejunnan25/pi05-robotwin"
            or manifest["task"] != "handover_block"
            or not re.fullmatch("[a-f0-9]{40}", manifest["revision"])):
        raise ValueError("invalid PACE download manifest")
    seen = {}
    for item in manifest["files"]:
        name = item["path"]
        path = PurePosixPath(name)
        if (name in seen or path.as_posix() != name or ".." in path.parts
                or (name != "tokenizer.model" and not name.startswith("handover_block/"))
                or item["bytes"] <= 0 or not re.fullmatch("[a-f0-9]{64}", item["sha256"])):
            raise ValueError(f"invalid manifest entry: {name}")
        seen[name] = item
    tokenizer = seen["tokenizer.model"]
    norm = seen["handover_block/assets/robotwin/norm_stats.json"]
    total = sum(item["bytes"] for item in seen.values())
    if (norm["path"] != manifest["norm_stats"] or norm["sha256"] != manifest["norm_stats_sha256"]
            or total != manifest["total_bytes"] or total - tokenizer["bytes"] != manifest["checkpoint_bytes"]):
        raise ValueError("manifest totals or normalization identity mismatch")
    return manifest


def local_path(root, name):
    path = root / name
    if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink in download path: {path}")
    return path


def verify_file(path, item):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError(f"missing file or size mismatch: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify_all(root, manifest):
    for item in manifest["files"]:
        verify_file(local_path(root, item["path"]), item)


def download_one(root, base_url, item):
    path = local_path(root, item["path"])
    if path.exists():
        verify_file(path, item)
        return
    partial = local_path(root, item["path"] + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not partial.exists() or partial.stat().st_size < item["bytes"]:
        # A pinned URL and curl's Range handling preserve interrupted transfers.
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
             "--speed-time", "90", "--speed-limit", "1", "--continue-at", "-",
             "--output", str(partial), base_url + quote(item["path"])],
            check=True,
        )
    verify_file(partial, item)
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path)
    parser.add_argument(
        "--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"),
        default="https://huggingface.co",
        help="download endpoint; the revision and SHA-256 checks apply to both",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    if not (args.download or args.verify):
        print(json.dumps(manifest, indent=2))
        return
    if args.dest is None:
        parser.error("--verify and --download require --dest")

    root = args.dest.resolve()
    if args.download:
        root.mkdir(parents=True, exist_ok=True)
        base = f'{args.endpoint}/{manifest["repo"]}/resolve/{manifest["revision"]}/'
        with local_path(root, ".pace-download.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for item in manifest["files"]:
                download_one(root, base, item)
    else:
        verify_all(root, manifest)
    print(f'Verified {len(manifest["files"])} files at {root}')


if __name__ == "__main__":
    main()
