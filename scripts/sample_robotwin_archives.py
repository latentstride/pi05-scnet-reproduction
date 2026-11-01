"""Read selected HDF5 members from official ZIPs using verified HTTP ranges.

No GPU or model dependency. A refused Range request aborts rather than fetching
an entire archive. ZIP CRC and extracted SHA-256 are recorded; the full archive
LFS SHA-256 is metadata, not a claimed full-download verification.
"""
import argparse
import hashlib
import io
import json
import re
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path


def context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, budget):
        self.url, self.size, self.budget = url, size, budget
        self.pos = self.transferred = self.requests = 0
        self.cache_start, self.cache = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if self.pos < 0:
            raise ValueError("Negative seek")
        return self.pos

    def read(self, count=-1):
        count = min(self.size - self.pos, count if count >= 0 else self.size)
        if count <= 0:
            return b""
        offset = self.pos - self.cache_start
        if not (0 <= offset and offset + count <= len(self.cache)):
            end = min(self.size - 1, self.pos + max(count, 65536) - 1)
            length = end - self.pos + 1
            if self.transferred + length > self.budget:
                raise RuntimeError("Transfer budget exceeded")
            request = urllib.request.Request(
                self.url + "?download=true&probe_range=" + str(self.pos) + "-" + str(end),
                headers={"Range": "bytes=%d-%d" % (self.pos, end), "User-Agent": "robotwin-data-probe/1"},
            )
            last_error = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, context=context(), timeout=45) as response:
                        expected = "bytes %d-%d/%d" % (self.pos, end, self.size)
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise RuntimeError("Server did not honor the exact HTTP Range")
                        payload = response.read(length + 1)
                    if len(payload) != length:
                        raise RuntimeError("Truncated HTTP Range")
                    self.cache_start, self.cache = self.pos, payload
                    self.transferred += length
                    self.requests += 1
                    break
                except Exception as exc:
                    # Do not print redirected signed URLs or credentials.
                    last_error = type(exc).__name__
                    if attempt == 2:
                        raise RuntimeError("Range fetch failed: " + last_error) from None
                    time.sleep(1)
            offset = 0
        data = self.cache[offset:offset + count]
        self.pos += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--budget-mb", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    listing = json.loads(args.metadata.read_text())
    records = []
    for robot in ["aloha-agilex", "arx-x5"]:
        item = next(x for x in listing if x["path"].endswith("/" + robot + "_clean_50.zip"))
        url = "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/" + args.revision + "/" + item["path"]
        remote = RangeFile(url, item["size"], args.budget_mb * 1024 * 1024)
        with zipfile.ZipFile(remote) as archive:
            infos = archive.infolist()
            members = [x for x in infos if x.filename.endswith((".hdf5", ".h5"))]
            members.sort(key=lambda x: [int(n) if n.isdigit() else n for n in re.split(r"(\d+)", x.filename)])
            if len(members) < args.episodes:
                raise RuntimeError("Insufficient HDF5 episodes; inspect member inventory")
            inventory = [{"name": x.filename, "bytes": x.file_size, "compressed_bytes": x.compress_size} for x in infos]
            (args.out / (robot + "-archive-members.json")).write_text(json.dumps(inventory, indent=2))
            selected = []
            for member in members[:args.episodes]:
                if member.file_size > 100 * 1024 * 1024:
                    raise RuntimeError("Individual member exceeds sample size limit")
                data = archive.read(member)  # zipfile verifies CRC on read.
                target = args.out / robot / Path(member.filename).name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(data)
                rec = {"member": member.filename, "file": robot + "/" + target.name,
                       "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                       "zip_crc32": "%08x" % member.CRC, "crc_verified": True}
                selected.append(rec)
                print("EXTRACTED", robot, target.name, len(data), flush=True)
        records.append({"robot": robot, "archive_path": item["path"], "archive_bytes": item["size"],
                        "archive_lfs_sha256_metadata": item["lfs"]["oid"],
                        "full_archive_sha256_verified": False, "hdf5_members": len(members),
                        "http_bytes_received": remote.transferred, "http_range_requests": remote.requests,
                        "selected": selected})
        (args.out / "sample_manifest.json").write_text(json.dumps({"repository": "TianxingChen/RoboTwin2.0",
            "revision": args.revision, "archives": records}, indent=2))
    print("SAMPLES_READY", flush=True)


if __name__ == "__main__":
    main()
