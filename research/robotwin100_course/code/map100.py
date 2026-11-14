"""Reuse frozen FK/tool_ik, never substitute a permissive collision checker.
Full recorded-command FK; deterministic IK subsample is geometry-only, not execution acceptance.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,csv,time,argparse,hashlib,concurrent.futures as cf
import numpy as np
R=Path(__file__).resolve().parents[1];PROJECT=R.parent
SCOPE=os.environ.get('IK_SCOPE','sampled');SUBDIR='mapping_full' if SCOPE=='full' else 'mapping';SUFFIX='_full' if SCOPE=='full' else ''
sys.path.insert(0,str(R/'sources/mapping_runtime'))
from g2_assets import load_arms
from g2_adapter import tool_ik,deadline,bounds,G2Error
from g2_kinematics import pose_error,BACKEND
from robotwin_control_geometry import pose_from_wxyz,inverse_pose

def setup():
 audit=R/'sources/embodiment_config_audit.json';smoke=json.loads((R/'sources/sapien_fk_smoke.json').read_text())
 config=json.loads((R/'sources/mapping_config.json').read_text());arms={};bases={}
 for robot in ['aloha-agilex','ARX-X5']:
  cal=config['calibration'][robot]
  for side in ['left','right']:
   ss=next(s for s in smoke['snapshots'] if s['robot']==robot and s['case']=='neutral' and any(a['side']==side for a in s['arms']))
   arm=next(a for a in ss['arms'] if a['side']==side);joint=next(j for j in ss['joints'] if j['name']==arm['ee_joint'])
   cal[side]['child_joint']=joint['pose_in_child'];cal[side]['id']='archived_static_smoke_child_frame:'+robot+':'+side
  arms[robot]=load_arms(R/'sources/embodiment_assets',audit,robot,cal)
  bases[robot]=[pose_from_wxyz([0,-.65,0],[np.sqrt(.5),0,0,np.sqrt(.5)])]*2 if robot=='aloha-agilex' else [pose_from_wxyz([x,-.35,.784],[np.sqrt(.5),0,0,np.sqrt(.5)]) for x in [-.3,.3]]
 return arms,bases

def job(arg):
 path,k=arg;row=json.loads(Path(path).read_text());key=f"{row['robot']}-episode{row['episode']}";dest=R/'data'/SUBDIR/f'{key}.json'
 if dest.exists():return json.loads(dest.read_text())
 robot='ARX-X5' if row['robot']=='arx-x5' else row['robot'];target='aloha-agilex' if robot=='ARX-X5' else 'ARX-X5';arms,bases=setup();a=np.load(R/'data/episodes'/f'{key}.npz')['state14'];n=len(a)
 selected=np.arange(n) if k==0 else np.unique(np.rint(np.linspace(0,n-1,min(k,n))).astype(int));seed=np.zeros((2,6));records=[];fk=[]
 for frame,state in enumerate(a):
  for side in [0,1]:
   q=state.reshape(2,7)[side,:6].astype(float);arm=arms[robot][side]
   try:
    bounds(q,arm.chain.limits);pose=bases[robot][side]@arm.chain.fk(q)@arm.offset();fk.append(dict(frame=frame,side=['left','right'][side],status='command_target_fk',xyz_m=pose[:3,3].tolist(),rotation=pose[:3,:3].tolist()))
   except Exception as e:fk.append(dict(frame=frame,side=['left','right'][side],status='source_position_limit' if isinstance(e,G2Error) else 'invalid_input',detail=str(e)))
 for frame in selected:
  for side in [0,1]:
   r=dict(source_robot=robot,target_robot=target,episode=row['episode'],frame=int(frame),side=['left','right'][side],collision_checked=False,execution_mapping_validated=False,fixture='nominal_G2_bases_not_historical_scene',state_semantics='recorded_command_target',status='not_attempted')
   try:
    q=a[frame].reshape(2,7)[side,:6].astype(float);bounds(q,arms[robot][side].chain.limits)
    tool=bases[robot][side]@arms[robot][side].chain.fk(q)@arms[robot][side].offset();gen=tool_ik(arms[target][side],bases[target][side],tool,seed[side],deadline(2.0))
    try:
     qt,diag=next(gen);bounds(qt,arms[target][side].chain.limits);error=pose_error(tool,bases[target][side]@arms[target][side].chain.fk(qt)@arms[target][side].offset());r.update(status='kinematic_solution',position_mm=float(np.linalg.norm(error[:3])*1000),angle_deg=float(np.rad2deg(np.linalg.norm(error[3:]))),target_q=qt.tolist(),target_tools_world=tool.tolist(),diagnostic=diag);seed[side]=qt
    except StopIteration as end:r.update(status='ik_failed',diagnostic=end.value,reason='finite_existing_seed_search_no_accepted_solution_not_global_unreachable')
   except G2Error as e:r.update(status='timeout' if e.status.value=='timeout' else 'source_position_limit',detail=str(e))
   except Exception as e:r.update(status='invalid_input',detail=type(e).__name__+':'+str(e))
   records.append(r)
 result=dict(robot=row['robot'],episode=row['episode'],n_frames=n,source_sha256=row['sha256'],fk_records=fk,ik_records=records,selected_frame_indices=selected.tolist(),full_execution_mapping='not_evaluated_missing_actual_state_historical_bases_and_collision_world')
 dest.write_text(json.dumps(result,separators=(',',':')));print(key,'IK',sum(r['status']=='kinematic_solution' for r in records),'/',len(records),flush=True);return result

def summarize():
 rows=[json.loads(p.read_text()) for p in (R/'data'/SUBDIR).glob('*.json')];rec=[x for r in rows for x in r['ik_records']];fk=[x for r in rows for x in r['fk_records']]
 keys=['source_robot','target_robot','episode','frame','side','status','position_mm','angle_deg','collision_checked','execution_mapping_validated','reason','detail']
 with (R/'data'/('ik_results'+SUFFIX+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r.get(k) for k in keys} for r in rec])
 total=[]
 for robot in ['aloha-agilex','ARX-X5']:
  rr=[x for x in rec if x['source_robot']==robot];ok=[x for x in rr if x['status']=='kinematic_solution'];fail={s:sum(x['status']==s for x in rr) for s in sorted(set(x['status'] for x in rr))}
  total.append(dict(source_robot=robot,target_robot='ARX-X5' if robot=='aloha-agilex' else 'aloha-agilex',attempted_arm_targets=len(rr),solutions=len(ok),solution_rate=len(ok)/len(rr) if rr else None,status_counts=fail,position_mm_max=max((x['position_mm'] for x in ok),default=None),position_mm_p95=float(np.percentile([x['position_mm'] for x in ok],95)) if ok else None,angle_deg_max=max((x['angle_deg'] for x in ok),default=None)))
 counts={s:sum(x['status']==s for x in fk) for s in sorted(set(x['status'] for x in fk))}
 summary=dict(trajectories=len(rows),all_fk_arm_targets=len(fk),fk_status_counts=counts,ik_arm_targets=len(rec),ik_frame_selection=('all original record indices; no outcome-based filtering' if SCOPE=='full' else '10 equispaced original indices per trajectory, fixed before outcomes; all attempts retained'),groups=total,backend=BACKEND,position_tolerance_mm=.5,angle_tolerance_deg=.05,full_execution_mapping_validated=False,unmet_conditions=['historical actual qpos not recorded','historical per-episode base poses not independently verified','full collision world and task contacts unavailable in raw HDF5','unknown original timestamps prevent velocity/acceleration checks'],historical_safety_gates_changed=False)
 (R/'data'/('mapping_summary'+SUFFIX+'.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2));(R/'data'/('mapping_details'+SUFFIX+'.json')).write_text(json.dumps(rows,separators=(',',':')));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--seconds',type=int,default=2400);p.add_argument('--frames',type=int,default=10);args=p.parse_args();(R/'data'/SUBDIR).mkdir(exist_ok=True);end=time.monotonic()+args.seconds
 while True:
  paths=[p for p in (R/'data/episodes').glob('*.json') if not p.name.endswith('-trajectory.json') and json.loads(p.read_text()).get('status')=='read_ok' and not (R/'data'/SUBDIR/p.name).exists()]
  with cf.ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(job,[(str(p),args.frames) for p in sorted(paths)]))
  s=summarize();print('MAPPING_COVERAGE',s['trajectories'],s['ik_arm_targets'],flush=True)
  if not args.watch or s['trajectories']==100 or time.monotonic()>end:break
  time.sleep(15)
