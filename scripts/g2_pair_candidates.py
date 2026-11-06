"""Assemble three diagnostic reset candidates from paired 10 g physics archives."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

def source_paths(root, episode):
    aloha = ('native-expert-episode0-20261106' if episode == 0
             else f'native-expert-rest-20261106/episode{episode}')
    return {'aloha-agilex': root / aloha / 'initial.json',
            'ARX-X5': root / 'arx-centered-20261106' / f'episode{episode}' / 'initial.json'}

def robot_state(record, robot):
    values = dict(zip(record['actual_all_joint_names'], record['actual_all_qpos'], strict=True))
    arms = {}
    for i, side in enumerate(('left', 'right')):
        native = robot == 'aloha-agilex'
        prefix = ('fl_' if i == 0 else 'fr_') if native else side + ':'
        joints = [values[prefix+f'joint{j}'] for j in range(1,9)]
        base = record['robot_base'] if native else record['robot_bases'][i]
        arms[side] = dict(base_world=base,q_rad=joints[:6],fingers_m=joints[6:])
    return arms

def generate(root, output):
    rows = []
    for episode in range(3):
        files = source_paths(root,episode)
        data = {robot:json.loads(path.read_text()) for robot,path in files.items()}
        aloha, arx = data.values()
        for field in ('block_pose','target_pose','block_mass_kg'):
            if aloha[field] != arx[field]:
                raise ValueError(f'episode {episode} mismatches on {field}')
        if abs(arx['block_mass_kg']-.01)>1e-8:
            raise ValueError('pair must use the 10 g task object')
        source = {k:dict(path=str(p.relative_to(root)),sha256=sha256(p.read_bytes()).hexdigest())
                  for k,p in files.items()}
        fixture = dict(task='handover_block', candidate_id=f'expert-reset-{episode}',
            formal_pair_id=None, source_reference_sha256=source['aloha-agilex']['sha256'],
            block_mass_kg=arx['block_mass_kg'],
            objects_world=dict(block=arx['block_pose'],target=arx['target_pose'],table=[0,0,.74,1,0,0,0]),
            robots={k:robot_state(v,k) for k,v in data.items()})
        rows.append(dict(seed=episode,episode=episode,fixture=fixture,source=source))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='diagnostic_candidates',candidates=rows),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('research',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    generate(args.research,args.output)
