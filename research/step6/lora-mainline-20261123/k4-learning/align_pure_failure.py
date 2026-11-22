"""Align saved policy outputs, approved endpoints, and measured physical TCP."""
import json
import sys
from pathlib import Path
import numpy as np
from openpi_client import msgpack_numpy
from g2_assets import load_arms
from g2_kinematics import pose_error

root,trial=map(Path,sys.argv[1:3])
bundle=root/'tools/a800-closed-loop'
safety=json.loads((bundle/'suite/docs/step6/arx_safety_profile.json').read_text())
arms=load_arms(bundle/'PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/assets',
 bundle/'pr5/research/g2/g2e-20261106-r1/release/asset-audit.json','ARX-X5',safety['calibration'])
records=[json.loads(x) for x in (trial/'trajectory.jsonl').read_text().splitlines()]
first=next(x for x in records if x['kind']=='physics')
bases=[np.asarray(first['tcp_world'])[i]@np.linalg.inv(arm.chain.fk(np.asarray(first['actual_qpos'])[i])@arm.offset()) for i,arm in enumerate(arms)]
def tcp(q):
 return np.asarray([bases[i]@arm.chain.fk(q[i*7:i*7+6])@arm.offset() for i,arm in enumerate(arms)])
def error(actual,target):
 e=pose_error(actual,target)
 return dict(mm=float(1000*np.linalg.norm(e[:3])),deg=float(np.degrees(np.linalg.norm(e[3:]))))
physics=None;requests={};responses={};attempts=[];completions=[];targets=[];active=None;query_stats={}
for row in records:
 kind=row['kind']
 if kind=='physics':physics=row
 elif kind=='policy_request':
  q=row['query_index'];requests[q]=row
  req=msgpack_numpy.unpackb((trial/row['payload']['file']).read_bytes())
  drive=tcp(np.asarray(req['state']))
  measured=np.asarray(physics['tcp_world'])
  query_stats[q]=dict(query=q,tick=physics['tick'],
   drive_actual_tcp=[error(measured[i],drive[i]) for i in range(2)],
   actual_left_tcp_world=measured[0,:3,3].tolist(),
   block_world=physics['block_pose_world'][:3],
   drive_state=np.asarray(req['state']).tolist(),actual_qpos=physics['actual_qpos'],
   actual_grip=physics['actual_gripper_aperture'])
 elif kind=='policy_response':
  q=row['query_index'];reply=msgpack_numpy.unpackb((trial/row['payload']['file']).read_bytes())
  pred=np.asarray(reply['actions']);responses[q]=pred
  stats=query_stats[q]
  stats['left_grip_h50']=pred[:,6].tolist()
  stats['first_close']={str(t):(int(h[0]) if len(h:=np.flatnonzero(pred[:,6]<=t)) else None) for t in (.9,.5,.1)}
  stats['left_tcp_k4_world']=[tcp(a)[0,:3,3].tolist() for a in pred[:4]]
 elif kind=='row_attempt':
  active=row;attempts.append(row)
 elif kind=='approved_row':
  goal=np.asarray(row['joint_goal']);req=np.asarray(row['input']).reshape(2,7)
  fg=[]
  for i,side in enumerate(('left','right')):
   cal=safety['calibration'][side];fg.append(np.asarray(cal['closed_m'])+req[i,6]*(np.asarray(cal['opened_m'])-cal['closed_m']))
  targets.append(dict(action=row['action_index'],arm_delta=float(np.max(abs(goal-req[:,:6]))),
   finger_delta=float(np.max(abs(np.asarray(row['finger_goal_m'])-fg)))))
 elif kind=='row_complete':
  actual=np.asarray(physics['tcp_world']);target=tcp(np.asarray(active['target']))
  completions.append(dict(action=row['action_index'],query=row['query_index'],
   tracking=[error(actual[i],target[i]) for i in range(2)]))
out=dict(queries=list(query_stats.values()),completed_tracking=completions,
 approved_endpoint_changes=targets,terminal_attempt=attempts[-1],
 stop=next((r for r in reversed(records) if r['kind']=='stop'),None),
 snapshot_scope='logged qpos, TCP and contacts; not a complete restorable PhysX snapshot')
(trial/'prediction-execution-alignment.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(queries=len(query_stats),last_query=query_stats[max(query_stats)],
 max_completed_left_tracking_mm=max((v['tracking'][0]['mm'] for v in completions),default=None),
 max_approved_arm_delta=max((v['arm_delta'] for v in targets),default=None),
 max_approved_finger_delta=max((v['finger_delta'] for v in targets),default=None),
 stop=out['stop']),indent=2))

