"""Read and verify the numeric and image artifacts of an archived GPU trial."""
from hashlib import sha256
import json

EPISODE = "unified-seed1-arx-full-w4"
JOBS = ("127999407", "127999421")


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return sha256(value).hexdigest()


class Trial:
    def __init__(self, root, job):
        self.job = job
        self.root = root / job
        self.episode = self.root / "trials" / EPISODE / "episode"
        self.manifest = read(self.root / "archive-manifest.json")
        self.release = dict((name, key) for key, name in
                            (line.split(maxsplit=1) for line in
                             (self.root / "release/SHA256SUMS").read_text().splitlines()))
        config = self.root / "release/configs" / f"{EPISODE}.json"
        self.config_sha = digest(config.read_bytes())
        if self.release[f"configs/{EPISODE}.json"] != self.config_sha:
            raise ValueError("trial configuration differs from release")
        self.config = read(config)

    def verified(self, path):
        record = self.manifest[str(path.relative_to(self.root))]
        raw = path.read_bytes()
        if digest(raw) != record["sha256"]:
            raise ValueError(f"archive hash mismatch {path.name}")
        return raw

    def events(self):
        path = self.episode / "events.jsonl"
        for line, raw in enumerate(self.verified(path).splitlines(), 1):
            yield line, json.loads(raw)
