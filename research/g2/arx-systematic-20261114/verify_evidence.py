"""Check arithmetic-analysis provenance and bounded CPU conclusions, without simulation."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def load(path):
    return json.loads(path.read_text())


def main():
    metrics=load(HERE/'chain-metrics.json')
    count=0
    for group in ('G1','G2','G3'):
        events=[json.loads(line) for line in
                (REPO/'research/g2/aloha-six-20261111'/group/'events.jsonl').read_text().splitlines()]
        responses=[e for e in events if e['kind']=='response']
        assert len(responses)==6
        for i,response in enumerate(responses):
            file=HERE/'source-evidence'/group/f'chunk-{i:02}.npy'
            assert sha256(file.read_bytes()).hexdigest()==response['chunk_sha256']
            assert np.load(file,allow_pickle=False).shape==(50,14)
            count+=1
        for observation in metrics['aloha'][group]['observations']:
            file=HERE/'source-evidence'/group/observation['file']
            assert sha256(file.read_bytes()).hexdigest()==observation['sha256']
        assert len(metrics['aloha'][group]['observations'])==6
    cpu={key:load(HERE/'cpu'/key/'receipt.json') for key in
         ('q2-online_boundary','q2-retain_checked_block','q1-retain_checked_block')}
    assert [cpu[k]['completed_rows'] for k in cpu]==[1,7,35]
    assert [cpu[k]['physical_steps'] for k in cpu]==[1,330,5710]
    for k,c in cpu.items():
        assert not c['stable_grasp'] and not c['holders'] and c['hold_no_advance']
        assert c['max_fingers'][0]<=1
        spec=load(HERE/'inputs'/f"{c['job']}-q{c['query']}.json")
        assert c['action_sha256']==spec['action_sha256']
        assert metrics['jobs'][c['job']]['query_summaries'][c['query']-1]['action_sha256']==c['action_sha256']
        if k!='q2-online_boundary':
            assert c['error']['status']=='collision'
            assert c['error']['diagnostic']['collision_world']=='target_collision'
            depths=[hit['depth_m'] for item in c['error']['diagnostic']['candidate_verdicts']
                    for hit in item.get('collisions',[]) if 'depth_m' in hit]
            assert depths and min(depths)>.002
    prior=load(REPO/'research/g2/arx-direct-online-20261114/results.json')['results']
    for job,r in metrics['jobs'].items():
        assert [q['action_sha256'] for q in r['query_summaries']]==[
            q['file_sha256'] for q in prior[job]['queries']]
        assert max(r['max_observation_round_trip_position_mm'])<.164
        assert max(r['max_target_ik_position_mm'])<.295
        assert min(r['min_measured_aperture_fraction'])>.994
        assert min(r['min_executed_gripper_drive_fraction'])>.994
        assert not r['input_dimensions_outside_training_q01_q99_by_more_than_0_001']
        assert r['query_summaries'][0]['gripper_threshold_first_rows']['0.99']==[49,None]
    assert load(REPO/'research/g2/arx-direct-cpu/expert-full.json')['success'] is True
    print(f'PASS: {count} original Aloha chunks; 18 input archives; 17 GPU action identities; '
          'FK/IK and gripper metrics; 3 bounded CPU receipts; expert reference')


if __name__=='__main__':
    main()
