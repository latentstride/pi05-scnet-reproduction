"""Download official pi05_libero files, resume transfers and verify byte lengths."""
import concurrent.futures
import json
import pathlib
import subprocess
import urllib.request

root = pathlib.Path(__file__).resolve().parents[1]
manifest_url = (
    'https://storage.googleapis.com/storage/v1/b/openpi-assets/o'
    '?prefix=checkpoints/pi05_libero/&fields=items(name,size),nextPageToken'
)
with urllib.request.urlopen(manifest_url, timeout=30) as response:
    manifest = json.load(response)
if manifest.get('nextPageToken'):
    raise RuntimeError('Manifest pagination requires handling before downloading')
(root / 'logs').mkdir(exist_ok=True)
(root / 'logs/checkpoint_manifest.json').write_text(json.dumps(manifest, indent=2))
base = root / 'cache/checkpoints/pi05_libero'


def fetch(item):
    name = item['name']
    rel = name.removeprefix('checkpoints/pi05_libero/')
    target = base / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item['size'])
    if target.exists() and target.stat().st_size == expected_size:
        return
    subprocess.run([
        'curl', '-fL', '--retry', '2', '--connect-timeout', '10', '--max-time', '1800',
        '-C', '-', 'https://storage.googleapis.com/openpi-assets/' + name, '-o', str(target),
    ], check=True)
    if target.stat().st_size != expected_size:
        raise RuntimeError(f'Incorrect size: {name}')
    print('OK', rel, expected_size, flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(fetch, manifest['items']))
print('ALL_CHECKPOINT_FILES_SIZE_VERIFIED', flush=True)
