"""Extract only planner and seed members from the fixed clean RoboTwin archives."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from sample_robotwin_archives import RangeFile

ROBOTS = ("aloha-agilex", "arx-x5")
REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"


class CurlRange(RangeFile):
    def read(self, count=-1):
        count = min(self.size - self.pos, count if count >= 0 else self.size)
        if count <= 0:
            return b""
        offset = self.pos - self.cache_start
        if not 0 <= offset <= len(self.cache) - count:
            end = min(self.size - 1, self.pos + max(count, 65536) - 1)
            size = end - self.pos + 1
            if self.transferred + size > self.budget:
                raise ValueError("planner transfer budget exhausted")
            data = subprocess.check_output([
                "curl", "--silent", "--show-error", "--fail", "--location",
                "--retry", "2", "--max-time", "35", "--max-filesize", str(max(size, 65536)),
                "--range", f"{self.pos}-{end}", "--write-out", "\n%{http_code}",
                f"{self.url}?download=true&native_range={self.pos}-{end}",
            ], timeout=120)
            payload, code = data.rsplit(b"\n", 1)
            if code != b"206" or len(payload) != size:
                raise ValueError("server did not return the requested byte count")
            self.cache_start, self.cache = self.pos, payload
            self.transferred += size
            self.requests += 1
            offset = 0
        self.pos += count
        return self.cache[offset:offset + count]


def extract(sample, output, robot, episode_limit):
    data = json.loads(sample.read_text())
    if data["revision"] != REVISION:
        raise ValueError("wrong fixed dataset revision")
    entry, = [row for row in data["archives"] if row["robot"] == robot]
    url = f"https://huggingface.co/datasets/{data['repository']}/resolve/{REVISION}/{entry['archive_path']}"
    remote = CurlRange(url, entry["archive_bytes"], 24 * 1024 * 1024)
    output.mkdir(parents=True, exist_ok=True)
    prefix = robot + "_clean_50/"
    members = {}
    with zipfile.ZipFile(remote) as archive:
        for key, name in [("seed.txt", "seed.txt"), *[
            (f"episode{i}.pkl", f"_traj_data/episode{i}.pkl") for i in range(episode_limit)]]:
            member = archive.getinfo(prefix + name)
            if member.file_size > 1_048_576:
                raise ValueError(f"unexpected metadata size: {name}")
            path = output / key
            if path.exists():
                import zlib
                payload = path.read_bytes()
                if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
                    raise ValueError(f"existing input differs from central directory: {path}")
            else:
                payload = archive.read(member)
                temporary = path.with_suffix(".partial")
                temporary.write_bytes(payload)
                temporary.replace(path)
            members[key] = dict(member=member.filename, bytes=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest(),
                                zip_crc32=f"{member.CRC:08x}")
    seeds = [int(value) for value in (output / "seed.txt").read_text().split()]
    if len(seeds) < 50 or len(set(seeds[:50])) != 50:
        raise ValueError("expected 50 unique original seeds")
    result = dict(robot=robot, revision=REVISION, archive_url=url,
                  archive_lfs_sha256_metadata=entry["archive_lfs_sha256_metadata"],
                  range_bytes=remote.transferred, members=members)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(robot, len(members), remote.transferred, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("robot", choices=ROBOTS)
    parser.add_argument("--episodes", type=int, choices=range(1, 51), default=50)
    args = parser.parse_args()
    extract(args.sample, args.output / args.robot, args.robot, args.episodes)
