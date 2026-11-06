#!/usr/bin/env python3
"""All-frame command-FK reachability screen; no collision or task-success claim."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import h5py
from g2_assets import load_arms
from g2_adapter import deadline
from g2_kinematics import pose_error
from g2_initial import matrix7
from robotwin_control_geometry import inverse_pose

def run(root,output):
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((root/'research/g2/expert-settle-source-mass-20261106/config.json').read_text())
    settled=json.loads((root/'research/g2/expert-settle-source-mass-20261106/settling.json').read_text())
    assets=root/'research/robotwin/embodiment_assets';audit=root/'research/robotwin/embodiment_config_audit.json'
    aa=load_arms(assets,audit,'aloha-agilex',config['calibration']['aloha-agilex'])
    rr=load_arms(assets,audit,'ARX-X5',config['calibration']['ARX-X5'])
    summaries=[]
    for ep,record in enumerate(settled['records']):
        a,r=(record['snapshots'][k]['initial'] for k in ('aloha-agilex','ARX-X5'))
        ab=[matrix7(a['robot_base_world'][s]) for s in ('left','right')]
        rb=[matrix7(r['robot_base_world'][s]) for s in ('left','right')]
        seeds=np.array([r['actual_arm_qpos_rad'][s] for s in ('left','right')]);cache=[{},{}];rows=[]
        with h5py.File(root/f'research/robotwin/data_probe/aloha-agilex/episode{ep}.hdf5') as f:qs=f['joint_action/vector'][:]
        began=time.monotonic()
        for frame,command in enumerate(qs):
            pair=[]
            for j,(source,target) in enumerate(zip(aa,rr)):
                source_q=command[:6] if j==0 else command[7:13];key=tuple(source_q)
                if key in cache[j]:
                    row=dict(cache[j][key],cached=True)
                else:
                    world=ab[j]@source.chain.fk(source_q)@source.offset()
                    goal=inverse_pose(rb[j])@world@inverse_pose(target.offset())
                    # Sum of all fixed translations bounds the terminal's root distance for any joint rotation.
                    radius=sum(np.linalg.norm(t[:3,3]) for t,axis in target.chain.joints)
                    distance=float(np.linalg.norm(goal[:3,3]))
                    row=dict(target_world_tcp=world.tolist(),root_terminal_distance_m=distance,outer_radius_bound_m=float(radius))
                    if distance>radius+1e-9:
                        row['status']='outside_conservative_outer_radius'
                    else:
                        stream=target.chain.ik(goal,seeds[j],deadline(1.),pos_tol=.0002,ang_tol=np.deg2rad(.01))
                        candidates=[];best=None
                        try:
                            while True:candidates.append(next(stream))
                        except StopIteration as end:best=end.value
                        except Exception as error:
                            row['search_error']=str(error)
                        if candidates:
                            q,res=min(candidates,key=lambda x:np.linalg.norm(x[0]-seeds[j]));seeds[j]=q
                            error=pose_error(world,rb[j]@target.chain.fk(q)@target.offset())
                            tcp=dict(position_m=float(np.linalg.norm(error[:3])),angle_deg=float(np.rad2deg(np.linalg.norm(error[3:]))))
                            row.update(status='ik_solution' if tcp['position_m']<=.0005 and tcp['angle_deg']<=.05 else 'tool_residual_failed',q=q.tolist(),residual=res,tcp_residual=tcp)
                        else:row.update(status='finite_ik_search_failed',best_residual=best)
                    cache[j][key]=row.copy()
                pair.append(row)
            rows.append(dict(frame=frame,arms=pair,both_ik_solved=all(x['status']=='ik_solution' for x in pair)))
        report=dict(episode=ep,frames=len(qs),wall_s=time.monotonic()-began,
                    both_ik_solved=sum(r['both_ik_solved'] for r in rows),
                    arm_status_counts=[dict(Counter(r['arms'][j]['status'] for r in rows)) for j in range(2)],records=rows)
        (output/f'episode{ep}.json').write_text(json.dumps(report,indent=2)+'\n')
        summaries.append({k:v for k,v in report.items() if k!='records'});print(summaries[-1],flush=True)
    result=dict(data_kind='diagnostic',input_semantics='FK_of_recorded_drive_targets_not_actual_expert_qpos',
                all_frames_screened=True,collision_checked=False,task_success_evaluated=False,formal_roster_frozen=False,
                note='Finite IK failure is not proof of unreachability; outside outer radius is a necessary geometric exclusion.',episodes=summaries)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
