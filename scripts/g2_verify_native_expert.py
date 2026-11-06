"""Verify dense CPU replay against immutable controls and held-out recorded endposes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from g2_kinematics import pose_error
from robotwin_control_geometry import pose_from_wxyz

def verify(root,output):
    folders=[root/'research/g2/native-expert-episode0-20261106']+[root/f'research/g2/native-expert-rest-20261106/episode{i}' for i in (1,2)]
    rows=[]
    for ep,folder in enumerate(folders):
        r=json.loads((folder/'replay.json').read_text());t=np.load(folder/'physics-trace.npz',allow_pickle=False)
        source=np.load(root/f'research/g2/expert-dense-episode{ep}-20261106/trace.npz',allow_pickle=False)
        n=len(source['target'])-1
        drives=np.c_[t['drive_native'][:n+1,:6],t['drive_native'][:n+1,8:14]]
        expected=np.c_[source['target'][:,:6],source['target'][:,7:13]]
        control_exact=bool(np.array_equal(drives,expected) and np.array_equal(t['normalized_gripper_target'][:n+1],source['target'][:,[6,13]]))
        frames=t['actual_endpose_matrix'][t['frame_ticks']];saved=source['expected_endpose']
        errors=np.array([[pose_error(pose_from_wxyz(v[:3],v[3:]),actual) for v,actual in zip(vs,poses)] for vs,poses in zip(saved,frames)])
        pos=float(np.linalg.norm(errors[:,:,:3],axis=2).max());angle=float(np.rad2deg(np.linalg.norm(errors[:,:,3:],axis=2)).max())
        contact=t['contacts'];initial=json.loads((folder/'initial.json').read_text());target=initial['target_pose'];top=(pose_from_wxyz(target[:3],target[3:])@np.array([0,0,.005,1]))[:3]
        predicate_consistent=True
        for tick,key in [(n,'native_success_at_expert_end'),(len(t['block_pose'])-1,'native_success_after_diagnostic_hold')]:
            v=t['block_pose'][tick];bottom=(pose_from_wxyz(v[:3],v[3:])@np.array([0,0,-.1,1]))[:3];delta=bottom-top
            verdict=bool(np.all(abs(delta[:2])<.03) and abs(delta[2])<.01 and t['normalized_gripper_target'][tick,1]>.8)
            predicate_consistent &= verdict==r[key]['success'] and bool(np.allclose(delta,r[key]['delta_m'],atol=1e-8,rtol=0))
        checks=dict(completed=r['completed'] and r['physics_steps_replayed']==n,
                    controls_exact=control_exact,predicate_matches_actual_trace=bool(predicate_consistent),
                    source_mass_matches=bool(abs(r['actual_block_mass_kg']-.01)<1e-8),
                    recorded_endpose_agrees=bool(pos<=.001 and angle<=.1),
                    physics_trace_finite=bool(all(np.isfinite(t[k]).all() for k in ('actual_native','drive_native','block_pose','contacts'))),
                    row_count=len(t['actual_native'])==n+1+200)
        rows.append(dict(episode=ep,seed=r['seed'],physics_steps=n,frames=len(frames),wall_s=r['wall_s'],checks=checks,verification_passed=all(checks.values()),
                         native_task_success=r['native_success_at_expert_end']['success'],hold_success=r['native_success_after_diagnostic_hold']['success'],
                         functional_point_delta_mm=(np.array(r['native_success_at_expert_end']['delta_m'])*1000).tolist(),
                         archived_ee_max_position_mm=pos*1000,archived_ee_max_angle_deg=angle,
                         max_finger_target_actual_penetration_mm=float(contact[:,7].max()*1000),
                         peak_finger_target_impulse_norm_sum_Ns=float(contact[:,3].max()),
                         peak_finger_target_impulse_over_dt_N=float(contact[:,3].max()/.004),
                         finger_target_nonzero_impulse_steps=int(np.count_nonzero(contact[:,3]>0)),
                         input_trace_sha256=r['trace_sha256'],physics_trace_sha256=hashlib.sha256((folder/'physics-trace.npz').read_bytes()).hexdigest(),folder=str(folder)))
    result=dict(data_kind='diagnostic',scope='3 fixed native Aloha expert reconstructions; no policy or ARX success rate',gpu_used=False,model_used=False,
                all_verified=all(r['verification_passed'] for r in rows),native_task_successes=sum(r['native_task_success'] for r in rows),episodes=rows)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.root,a.output)
