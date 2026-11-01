"""Verify GCS checkpoint metadata (CRC32C plus MD5 when present)."""
import argparse
import base64
import hashlib
import json
import pathlib
import urllib.request

import google_crc32c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=pathlib.Path, default=pathlib.Path('logs/checkpoint_manifest_verified.json'))
    p.add_argument('--root', type=pathlib.Path, default=pathlib.Path('cache/checkpoints/pi05_libero'))
    p.add_argument('--output', type=pathlib.Path, default=pathlib.Path('results/checkpoint_verification.json'))
    args = p.parse_args()
    if not args.manifest.exists():
        url = ('https://storage.googleapis.com/storage/v1/b/openpi-assets/o'
               '?prefix=checkpoints/pi05_libero/&fields=items(name,size,md5Hash,crc32c,generation),nextPageToken')
        with urllib.request.urlopen(url, timeout=30) as response:
            manifest = json.load(response)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2))
    else:
        manifest = json.loads(args.manifest.read_text())
    assert not manifest.get('nextPageToken'), 'Manifest is incomplete: additional page exists'
    prefix = 'checkpoints/pi05_libero/'
    reports = []
    for item in manifest['items']:
        relative = pathlib.PurePosixPath(item['name'].removeprefix(prefix))
        assert item['name'].startswith(prefix) and '..' not in relative.parts
        path = args.root / relative
        assert path.stat().st_size == int(item['size']), path
        crc = google_crc32c.Checksum()
        md5 = hashlib.md5()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                crc.update(chunk)
                md5.update(chunk)
        assert base64.b64encode(crc.digest()).decode() == item['crc32c'], path
        if item.get('md5Hash'):
            assert base64.b64encode(md5.digest()).decode() == item['md5Hash'], path
        reports.append({'name': item['name'], 'size': int(item['size']), 'generation': item.get('generation'),
                        'crc32c_verified': True, 'md5_verified': bool(item.get('md5Hash'))})
        print('VERIFIED', item['name'], flush=True)
    assert len(reports) == 16, len(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))
    print('ALL_CHECKPOINT_HASHES_VERIFIED', len(reports), flush=True)


if __name__ == '__main__':
    main()
