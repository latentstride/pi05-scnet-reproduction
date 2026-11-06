"""Freeze a concrete two-pair G2 release in an existing staging directory."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

CAPS = {'A-native-w1':1250, 'A-native-w4':950, 'B-arx-w1':2600, 'B-arx-w4':2600}
STAGE_CAPS = {'dynamic':1850, 'aloha':2530, 'arx':5230, 'ablation':7160, 'expert':560, 'statistics':120}
# 15,000 s of trial caps including process margins, 1,850 s of startup,
# 560 s for expert evidence, 120 s statistics, and 300 s graphics.
TOTAL_S = 18000
GRAPHICS_S = 300
SUPERVISOR_S = sum(STAGE_CAPS.values())
assert GRAPHICS_S + SUPERVISOR_S + 60 <= TOTAL_S


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def prepare(release, candidate_dir, checkpoint_receipt):
    checkpoint_manifest = release/'pace_download_manifest.json'
    model = json.loads(checkpoint_manifest.read_text())
    checkpoint = dict(id=model['config'],revision=model['revision'],
                      manifest_sha256=digest(checkpoint_manifest),norm_stats_sha256=model['norm_stats_sha256'])
    audit = json.loads((release/'asset-audit.json').read_text())
    robots = {r['robot']:r for r in audit['robots']}
    source = json.loads((release/'source-audit.json').read_text())
    scene_hash = source['envs/handover_block.py']
    probes = {r['episode']:r for r in json.loads((candidate_dir/'pair-probe.json').read_text())}
    # Lock in source order, before any G2 policy outcomes. Seed 2 is held in reserve.
    seeds = (0,1)
    (release/'configs').mkdir()
    episodes, bindings = [], {}
    for seed in seeds:
        probe = probes[seed]
        if (probe['objects_mismatched'] or not probe['arx_first_observation']['ok']
                or not all(r['clear'] for r in probe['geometry'].values())):
            raise ValueError(f'candidate {seed} failed CPU preflight')
        for condition, timeout in CAPS.items():
            episode = f'seed{seed}-{condition}'
            config = json.loads((candidate_dir/'startup'/f'{episode}.json').read_text())
            native = config['embodiment']=='aloha-agilex'
            pair = f'paired-10g-{seed:02d}'
            fixture = config['reset_fixture']
            fixture['formal_pair_id'] = pair
            cfg_hash = sha256(json.dumps(fixture,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            expected = deepcopy(probe['actual_initial'][config['embodiment']])
            expected['reset_fixture_sha256'] = cfg_hash
            config.update(max_actions=300,timeout_s=timeout,budget_scope='frozen trial wall and action caps')
            config_path = release/'configs'/f'{episode}.json'
            write(config_path,config)
            stable_config = {k:v for k,v in config.items() if k not in ('reset_fixture','scene_seed')}
            config_protocol_hash = sha256(json.dumps(stable_config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            row = dict(episode_id=episode,task='handover_block',seed=seed,
                       comparison_id=f'transfer-w{config["window"]}',pair_id=pair,
                       condition=dict(id=condition,embodiment='aloha-agilex' if native else 'arx-x5',
                           action_source='source_policy' if native else 'target_policy',checkpoint=checkpoint,
                           expert_representation=None,execution='qpos',window_actions=config['window'],
                           config_sha256=config_protocol_hash,observation_mapping='native' if native else 'arx_proxy_v1',
                           action_mapping='native' if native else 'arx_quintic_contact_v1'),
                       environment=dict(runtime_id='robotwin-cu122-scnet',
                           robotwin_revision='PACE-vendored-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1',
                           pace_revision='fd2e624cc19325cc9b20a620c82d51c0d8e25cc1',
                           assets_revision=digest(release/'asset-audit.json'),scene_config_sha256=scene_hash,
                           embodiment_config_sha256=robots[config['embodiment']]['config_sha256']))
            episodes.append(row)
            bindings[episode] = dict(config_sha256=digest(config_path),expected_initial=expected,
                                     trajectory_reference_id='per_tick_drive_tracking_v1')
    write(release/'run_lock.json',dict(schema_version=1,status='frozen',episodes=episodes,bindings=bindings,
          artifacts={name:digest(release/f'{name.replace("_","-")}.json') for name in ('asset_audit','source_audit')}))
    write(release/'launch-spec.json',dict(supervisor_timeout_s=SUPERVISOR_S,graphics_timeout_s=GRAPHICS_S,
          stage_timeout_s=STAGE_CAPS,trial_timeout_s=CAPS,approved_allocation_s=TOTAL_S,approved_node='gnode39',
          model_cpu_receipt=str(checkpoint_receipt),model_cpu_receipt_sha256=digest(checkpoint_receipt),
          seed_selection='first two prequalified archive episodes in source order, before policy evaluation',
          expert_scope='one live seed-0 original 10g CPU reference with centered grip and continuous contact guard'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('release','candidate-dir','checkpoint-receipt'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    prepare(a.release,a.candidate_dir,a.checkpoint_receipt)
