"""Build diagnostic bring-up configs from CPU-verified reset candidates."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from g2_initial import digest_json
from g2_task_runtime import validate_config

PROMPT = 'Pick up the red block with the left arm, hand it to the right arm, and place it on the target.'

def build_configs(root, output):
    evidence = root/'pair-candidates-20261106'
    candidates = json.loads((evidence/'fixtures.json').read_text())['candidates']
    probe_path = evidence/'pair-probe.json'
    probes = {r['episode']:r for r in json.loads(probe_path.read_text())}
    rows = []
    output.mkdir(parents=True,exist_ok=False)
    for candidate in candidates:
        fixture = candidate['fixture']
        seed = candidate['seed']
        result = probes[candidate['episode']]
        if (result['fixture_sha256'] != digest_json(fixture) or result['objects_mismatched']
                or not result['arx_first_observation']['ok']
                or not all(v['clear'] and v['joint_readback_max_rad'] <= 1e-6
                           for v in result['geometry'].values())):
            raise ValueError(f'candidate {seed} did not pass reset verification')
        for robot, label in (('aloha-agilex','A-native'),('ARX-X5','B-arx')):
            for window in (1,4):
                template = (root/f'diagnostic-configs-20261105/{label}-w{window}.json' if robot=='aloha-agilex'
                            else root/f'contact-configs-centered-20261106/{label}-w{window}-centered.json')
                cfg = json.loads(template.read_text())
                source = fixture['robots']['aloha-agilex']
                cfg.update(reset_fixture=deepcopy(fixture),scene_seed=seed,episode_info={},
                    source_base_world=[source[side]['base_world'] for side in ('left','right')],
                    prompt=PROMPT, max_actions=4,timeout_s=120.,
                    budget_scope='diagnostic startup only; formal episode duration and step ceiling pending')
                validate_config(cfg)
                path = output/f'seed{seed}-{label}-w{window}.json'
                path.write_text(json.dumps(cfg,indent=2)+'\n')
                rows.append(dict(seed=seed,episode_id=path.stem,condition=cfg['condition'],
                    embodiment=robot,window_actions=window,config=path.name,
                    config_sha256=sha256(path.read_bytes()).hexdigest(),
                    candidate_id=fixture['candidate_id'],formal_pair_id=None))
    manifest = dict(status='diagnostic_startup_only',formal_run_lock_frozen=False,
                    cpu_pair_probe_sha256=sha256(probe_path.read_bytes()).hexdigest(),
                    prompt=PROMPT,episodes=rows)
    (output/'diagnostic-roster.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('research',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    build_configs(args.research,args.output)
