"""Bind the CPU replay to the archived source requests and measured fixture states."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import numpy as np

EPISODE='unified-seed1-arx-full-w4'


def load(path):return json.loads(path.read_bytes())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def assemble(raw,cpu,repo,out):
    source=raw/'results/pace-g2-standalone-arx-127510363/trials'/EPISODE/'episode'
    report=repo/'research/g2/job-127510363/result.json'
    data=load(report)['source_sha256']; initial=load(source/'result.json')['initial']
    hashes=load(cpu/'CPU-SHA256.json')
    assert all(sha256((cpu/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    replay_path=next(cpu.glob('results/g2/*/comparison.json'))
    replay=load(replay_path)
    requests={};last=None;count=0
    with (source/'events.jsonl').open('rb') as stream:
        for line in stream:
            row=json.loads(line)
            if row['kind']=='executed_step':last=row,sha256(line).hexdigest()
            if row['kind']=='policy_request':
                count+=1
                if count in (35,36):
                    assert row['stamp']['tick']==last[0]['tick']
                    action=source/f'action-{count:05d}.npy'
                    digest=sha256(action.read_bytes()).hexdigest()
                    assert digest==data[f'trials/{EPISODE}/episode/{action.name}']
                    requests[count]=dict(request=row,request_raw_line_sha256=sha256(line).hexdigest(),
                                         measured=last[0],measured_raw_line_sha256=last[1],
                                         action_sha256=digest,action_shape=[50,14],
                                         raw_action=np.load(action,allow_pickle=False).tolist())
                    if count==36:break
    write(out/'arx-input-snapshots.json',dict(initial=initial,requests=requests,source_sha256=data))
    compact={}
    for r in replay['reports']:
        q=r['query']; path=replay_path.parent/str(q)/'new/dynamic-events.jsonl'
        rows=[json.loads(line) for line in path.open()]
        assert all(x.get('physics_step_count')==1 and x['tick']==r['tick']+i for i,x in enumerate(rows,1))
        assert len(rows)==r['dynamic']['physics_steps']
        phases=sorted({x['contact_state']['phase'] for x in rows})
        assert phases==['approach'] and all(not x['contact_state']['holders'] for x in rows)
        tracked={side:sum(bool(x['contact_state']['contact_fingers'][side]) for x in rows) for side in ('left','right')}
        path_npz=replay_path.parent/str(q)/'new/plan.npz'
        with np.load(path_npz,allow_pickle=False) as plan:
            waypoint_ticks=plan['waypoint_ticks'].tolist()
            waypoints=[rows[round(t*float(plan['period_s'])/.004)-1] for t in waypoint_ticks]
        attempts=r['new']['receipt']['attempts']; errors=attempts[-1]['errors']
        compact[q]=dict(waypoint_ticks=waypoint_ticks,physical_phase_counts=r['dynamic']['phases'],
                        measured_finger_contact_ticks=tracked,
                        max_nominal_tcp_position_change_m=np.max([e['position_m'] for e in errors],axis=0).tolist(),
                        max_nominal_tcp_angle_change_deg=np.max([e['angle_deg'] for e in errors],axis=0).tolist(),
                        max_physics_joint_tracking_error_rad=float(max(np.max(abs(np.asarray(x['actual_arm'])-x['target_arm'])) for x in rows)),
                        waypoint_readbacks=waypoints)
    result35=replay['reports'][0]['dynamic']; historical36=requests[36]['measured']
    a=np.asarray(result35['final_tools'])[:, :3, 3];b=np.asarray(historical36['actual_tools_task'])[:, :3, 3]
    difference=dict(target_tcp_position_error_m=np.linalg.norm(a-b,axis=1).tolist(),
                    max_joint_error_rad=float(np.max(abs(np.asarray(result35['final_q'])-historical36['actual_arm']))),
                    state_reuse_allowed=False)
    ax=np.asarray(initial['objects_task']['block'][:3]);at=np.asarray(initial['objects_task']['target'][:3]); init=[]
    for g in ('G1','G2','G3','G4','G5'):
        p=repo/'research/g2/aloha-six-20261111'/g/'initial-physics.json';r=load(p)
        xyz=np.asarray(r['block_pose'])[:3,3];t=np.asarray(r['target_pose'])[:3,3]
        init.append(dict(experiment=g,block_xyz_m=xyz.tolist(),target_xyz_m=t.tolist(),
                         block_arx_delta_xyz_m=(xyz-ax).tolist(),target_arx_delta_xyz_m=(t-at).tolist(),
                         block_target_xy_spacing_m=float(np.linalg.norm((t-xyz)[:2])),block_mass_kg=r['block_mass_kg'],
                         robot_root=r['robot_root'],tick0_sha256=sha256(p.read_bytes()).hexdigest()))
    write(out/'comparison-readback.json',dict(waypoints=compact,corrected_35_end_vs_historical_36=difference,
          initial_comparison=dict(aloha=init,arx_block_target_xy_spacing_m=float(np.linalg.norm((at-ax)[:2])),
          arx_initial=initial,physical_pairing=False),
          gpu_gate='NOT_READY',blocker='No validated post-intervention strategy transition into a safe left grasp under ARX contact constraints. The next historical chunk is tied to a different state.'))
    shutil.copyfile(replay_path,out/'comparison.json')
    shutil.copyfile(cpu/'CPU-SHA256.json',out/'cpu-source-sha256.json')
    shutil.copyfile(next(cpu.glob('tools/*/cpu.log')),out/'cpu.log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('raw','cpu','repo','out'):p.add_argument(arg,type=Path)
    a=p.parse_args();assemble(a.raw,a.cpu,a.repo,a.out)
