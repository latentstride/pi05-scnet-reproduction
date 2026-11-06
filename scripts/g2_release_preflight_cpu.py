"""Renderer-free validation of the exact G2 release bundle and prior model receipt."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

from g2_locked_task import load_trial
from g2_task_runtime import inspect_assets, verify_runtime
from summarize_experiments import check_formal_roster


def verify(release, root):
    source_audit = json.loads((release/'source-audit.json').read_text())
    pace = root/'source/PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1'
    verify_runtime(pace,source_audit)
    lock = release/'run_lock.json'
    lock_sha = sha256(lock.read_bytes()).hexdigest()
    roster = json.loads(lock.read_text())['episodes']
    schema_fixture = [dict(row,run_lock_sha256=lock_sha) for row in roster]
    check_formal_roster(lock,schema_fixture)
    for row in roster:
        episode = row['episode_id']
        config,_,_ = load_trial(lock,lock_sha,episode,release/'configs'/f'{episode}.json')
        inspect_assets(config,pace/'assets',release/'asset-audit.json')
    spec = json.loads((release/'launch-spec.json').read_text())
    receipt = Path(spec['model_cpu_receipt'])
    if sha256(receipt.read_bytes()).hexdigest() != spec['model_cpu_receipt_sha256']:
        raise ValueError('model receipt drift')
    manifest = json.loads((release/'pace_download_manifest.json').read_text())
    files = {}
    for item in manifest['files']:
        path = root/'cache/pace-handover'/item['path']
        stat = path.stat()
        if path.resolve() != path or stat.st_size != item['bytes']:
            raise ValueError(f'model path or byte count changed: {item["path"]}')
        files[item['path']] = [stat.st_size,stat.st_mtime_ns,item['sha256']]
        if item['path'] in ('tokenizer.model',manifest['norm_stats']):
            if sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('tokenizer or normalization hash changed')
    prior = json.loads(receipt.read_text())
    if (prior['status']!='cpu_preflight_passed' or prior['checkpoint_revision']!=manifest['revision']
            or prior['checkpoint_files']!=files):
        raise ValueError('checkpoint no longer matches the prior full CPU SHA-256 verification')
    tokenizer = root/'cache/openpi-pace/big_vision/paligemma_tokenizer.model'
    if sha256(tokenizer.read_bytes()).hexdigest()!=files['tokenizer.model'][2]:
        raise ValueError('runtime tokenizer changed')
    return dict(passed=True,scope='CPU only; assets, roster and model receipt',run_lock_sha256=lock_sha,
                trial_count=len(roster),model_files_bound_to_prior_sha256_receipt=len(files))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('release','root','output'):
        p.add_argument(n,type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(verify(a.release,a.root),indent=2)+'\n')
