"""Fetch official gitlink revisions when git clone is unavailable (no Git history)."""
import json
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
sha = (root / 'source/OPENPI_COMMIT').read_text().strip()
src = root / 'source' / ('openpi-' + sha)
repos = {
    'third_party/aloha': 'Physical-Intelligence/aloha',
    'third_party/libero': 'Lifelong-Robot-Learning/LIBERO',
}
for entry in json.loads((root / 'logs/source_tree.json').read_text())['tree']:
    if entry['mode'] != '160000':
        continue
    path, rev = entry['path'], entry['sha']
    if path not in repos:
        raise RuntimeError(f'Unexpected submodule: {path}')
    archive = root / 'source' / (path.split('/')[-1] + '-' + rev + '.tar.gz')
    subprocess.run([
        'curl', '-fL', '--retry', '2', '--max-time', '180',
        f'https://codeload.github.com/{repos[path]}/tar.gz/{rev}', '-o', str(archive),
    ], check=True)
    target = src / path
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '-xzf', str(archive), '--strip-components=1', '-C', str(target)], check=True)
    print(path, rev, flush=True)
