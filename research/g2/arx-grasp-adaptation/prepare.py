"""Extract G1 query 2 and its exact same-tick real snapshot."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import tarfile

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
BATCH = STUDY / "arx-online-batch-20261115"
EPISODE = "unified-seed1-arx-full-w4"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def prepare(root, output):
    relative = Path("pace-g2-standalone-arx-128047752/trials") / EPISODE / "episode"
    folder = root / relative
    source = json.loads((BATCH / "results/G1/members.json").read_text())
    for name in ("action-00002.npy", "events.jsonl", "result.json"):
        if digest(folder / name) != source[str(relative / name)]:
            raise ValueError(f"{name} differs from the published manifest")
    candidate = STUDY / "arx-stage6-concentrated-20261114/gpu-candidates/G1/g2-standalone-arx.tar.gz"
    with tarfile.open(candidate) as archive:
        config = json.load(archive.extractfile(f"g2-standalone-arx/configs/{EPISODE}.json"))
    count = 0
    previous = None
    with (folder / "events.jsonl").open() as stream:
        for event in map(json.loads, stream):
            if event.get("physics_step_count"):
                previous = event
            if event["kind"] == "policy_request":
                count += 1
                if count == 2:
                    if previous is None or previous["tick"] != event["stamp"]["tick"]:
                        raise ValueError("no same-tick measured scene")
                    break
        else:
            raise ValueError("G1 query 2 is missing")
    initial = json.load((folder / "result.json").open())["initial"]
    chunk = np.load(folder / "action-00002.npy", allow_pickle=False)
    if chunk.shape != (50, 14) or not np.isfinite(chunk).all():
        raise ValueError("invalid archived policy output")
    output.write_text(json.dumps(dict(job="128047752", query=2, request=event, config=config,
        initial=initial, at=previous, chunk=chunk.tolist(),
        action_sha256=source[str(relative / "action-00002.npy")]), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.output)
