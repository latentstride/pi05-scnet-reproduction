from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
S=json.loads((R/'data/summary.json').read_text());T=json.loads((R/'data/trajectories.json').read_text())
Q=list(csv.DictReader((R/'data/trajectory_quality.csv').open()));mp=R/'data/mapping_summary_full.json';M=json.loads((mp if mp.exists() else R/'data/mapping_summary.json').read_text())
# Full mapping is loaded when completed; never label a partial full run as finished.
full=M['trajectories']==100 and M['ik_frame_selection'].startswith('all')
F=list(csv.DictReader((R/'data/ik_results_full.csv' if full else R/'data/ik_results.csv').open()))
def hist(v,bins=25):
 v=np.asarray(v,dtype=float);v=v[np.isfinite(v)];counts,edges=np.histogram(v,bins=bins);return dict(counts=counts.tolist(),edges=edges.tolist(),n=int(len(v)))
stats={}
for robot in ['aloha-agilex','arx-x5']:
 tt=[t for t in T if t['robot']==robot];a=np.vstack([np.load(R/'data/episodes'/f"{robot}-episode{t['episode']}.npz")['state14'] for t in tt]);ee=np.vstack([np.load(R/'data/episodes'/f"{robot}-episode{t['episode']}.npz")['left_endpose7'] for t in tt]);d=np.vstack([np.diff(np.load(R/'data/episodes'/f"{robot}-episode{t['episode']}.npz")['state14'],axis=0) for t in tt])
 stats[robot]=dict(frame_lengths=[dict(episode=t['episode'],frames=t['n']) for t in tt],state_ranges=[dict(dim=i,min=float(a[:,i].min()),max=float(a[:,i].max()),mean=float(a[:,i].mean()),hist=hist(a[:,i])) for i in range(14)],left_xyz_mm=[hist(ee[:,i]*1000) for i in range(3)],left_quaternion_wxyz=[hist(ee[:,i]) for i in range(3,7)],delta_q_per_record=[hist(d[:,i]) for i in range(14)],delta_semantics='per adjacent original record, not a velocity; no known dt')
 right=np.vstack([np.load(R/'data/episodes'/f"{robot}-episode{t['episode']}.npz")['right_endpose7'] for t in tt])
 stats[robot]['right_xyz_mm']=[hist(right[:,i]*1000) for i in range(3)]
 stats[robot]['right_quaternion_wxyz']=[hist(right[:,i]) for i in range(3,7)]
 old=robot if robot=='aloha-agilex' else 'ARX-X5';rr=[r for r in F if r['source_robot']==old];ok=[r for r in rr if r['status']=='kinematic_solution'];stats[robot]['ik_error_hist']=dict(position_mm=hist([float(r['position_mm']) for r in ok]),angle_deg=hist([float(r['angle_deg']) for r in ok]));stats[robot]['ik_failures']=[r for r in rr if r['status']!='kinematic_solution']
(R/'data/distributions.json').write_text(json.dumps(stats,separators=(',',':')))
# Shared historical evidence retained independently of the new expert data.
cpu=list(csv.DictReader((R/'sources/contact_curves.csv').open()))
for r in cpu:
 for k in ['tick','time_since_hold_s','slip_mm','drive_left_mm','block_z_mm']:r[k]=float(r[k])
research=dict(libero=dict(checkpoint='pi05_libero',trials=15,success=15),unified_127147943=dict(policy_trials=8,policy_success=0,repaired_expert_trials=2,repaired_expert_success=2),latest_127192740=dict(planned=10,run=1,not_run=9,policy_success=0,complete_actions=27,model_queries=2,source_commit='0467a96258a2e3982196ae928a153ed85bddce8d'),preload=dict(scope='CPU H50 seed1, one branch per intervention; not model closed-loop success',gap1='contact lost at tick5090',lift_mm=99.956453,min_margin_mm=.587928,max_slip_mm=4.412072),cpu_rows=cpu)
(R/'data/research_results.json').write_text(json.dumps(research,ensure_ascii=False,indent=2))
payload=dict(summary=S,quality=Q,trajectories=T,distributions=stats,mapping=M,research=research,full_geometric_ik=full,sources=dict(revision=S['source_revision'],pr5='0467a96258a2e3982196ae928a153ed85bddce8d',download_manifest='data/download_manifest.json',statistics='data/*.csv and *.json'))
content=json.dumps(payload,ensure_ascii=False,separators=(',',':'));(R/'data/dashboard_data.json').write_text(content);(R/'dashboard/data.js').write_text('window.ANALYSIS='+content+';\n')
print(json.dumps({'trajectories':len(T),'frames':S['total_frames'],'geometric_ik_arm_targets':M['ik_arm_targets'],'full_geometric_ik':full,'dashboard_bytes':len(content.encode())},ensure_ascii=False))
