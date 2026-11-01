"""Resume official checkpoint files in bounded HTTP byte ranges.

Run after download_checkpoint.py has saved logs/checkpoint_manifest.json.
Do not run another downloader against the same files concurrently.
"""
import concurrent.futures
import json
import pathlib
import time
import urllib.parse

import requests

root = pathlib.Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'logs/checkpoint_manifest.json').read_text())
if manifest.get('nextPageToken'):
    raise RuntimeError('Manifest is incomplete; handle pagination first')
base = root / 'cache/checkpoints/pi05_libero'


def fetch(item):
    name, size = item['name'], int(item['size'])
    target = base / name.removeprefix('checkpoints/pi05_libero/')
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        'https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/'
        + urllib.parse.quote(name, safe='') + '?alt=media',
        'https://storage.googleapis.com/openpi-assets/' + name,
    ]
    with requests.Session() as session:
        while (target.stat().st_size if target.exists() else 0) < size:
            start = target.stat().st_size if target.exists() else 0
            end = min(start + 8 * 1024 * 1024, size) - 1
            for attempt in range(8):
                try:
                    with session.get(
                        urls[attempt % 2], headers={'Range': f'bytes={start}-{end}',
                                                   'Accept-Encoding': 'identity'},
                        timeout=(8, 30), stream=True,
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise RuntimeError('Range not honored')
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                            raise RuntimeError('Incorrect range')
                        chunk = response.content
                        if len(chunk) != end - start + 1:
                            raise RuntimeError('Short range')
                    with target.open('ab') as output:
                        output.write(chunk)
                    print('PROGRESS', target.name, end + 1, size, flush=True)
                    break
                except Exception as exc:
                    print('RETRY', target.name, attempt, type(exc).__name__, flush=True)
                    time.sleep(min(2 ** attempt, 15))
            else:
                raise RuntimeError('Range retries exhausted: ' + name)
    if target.stat().st_size != size:
        raise RuntimeError('Size mismatch')
    print('COMPLETE', target.name, size, flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(fetch, manifest['items']))
print('ALL_FILES_COMPLETE', flush=True)
