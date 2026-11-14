import argparse
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
from robotwin_control_geometry import pose_from_wxyz
parser=argparse.ArgumentParser(description='Bind native control and GPU contact-window inputs to source hashes')
parser.add_argument('prepared',type=Path)
parser.add_argument('gpu_archive',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
d=Path('research/g2/arx-stage6-concentrated-20261114/inputs')
template=json.loads((d/'q2-block8.json').read_text())
out=args.output
out.mkdir()
base=Path('research/g2/native-expert-100/results/arx-x5')
for episode in (1,7):
 ep=f'episode{episode:02}'
 root=base/ep
 receipt=json.loads((root/'receipt.json').read_text())
 prepared=args.prepared/'arx-x5'/ep/'controls.npz'
 trace=root/'attempt/physics-trace.npz'
 for path,expected in ((prepared,receipt['input']['controls_sha256']),(trace,receipt['physics_trace_sha256'])):
  if sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(path)
 start=receipt['input']['segments'][2]['first_tick']
 finish=receipt['input']['segments'][3]['last_tick']
 with np.load(trace) as t,np.load(prepared) as c:
  indices=np.unique(c['frame_ticks'][(c['frame_ticks']>=start)&(c['frame_ticks']<=finish)])
  # Original controller waypoints at actual archive frames, no expert pose shifts;
  # preserve the cached closure progression and first lift phase.
  q=t['actual_qpos'][start].reshape(2,8)
  b=t['block_pose'][start].tolist()
  start_f=t['native_drive'][start].reshape(2,8)[:,6:]
  native=t['contacts']
  first=int(np.flatnonzero((native[:,0].astype(int)&3)!=0)[0])
  bilateral=int(np.flatnonzero((native[:,0].astype(int)&3)==3)[0])
  closure=t['normalized_gripper_request'][first,0]
  nominal=dict(start_tick=start,finish_tick=finish,indices=indices[1:].tolist(),
   first_left_contact_tick=first,first_left_bilateral_tick=bilateral,first_lift_tick=receipt['contacts']['left_bilateral_lift'],
   normalized_left_request_at_first_contact=float(closure),
   block_shift_before_bilateral_mm=(1000*(t['block_pose'][bilateral,:3]-t['block_pose'][start,:3])).tolist(),
   left_fingers_at_bilateral_mm=(1000*t['actual_qpos'][bilateral,6:8]).tolist(),
   terminal_block_pose=t['block_pose'][finish].tolist(),native_rows=c['target'][indices[1:]].tolist())
  tcp=t['actual_endpose_matrix'][first,0]
  block=t['block_pose'][first,:3]
  nominal['first_block_in_tool_mm']=(1000*tcp[:3,:3].T@(block-tcp[:3,3])).tolist()
  init=json.loads(json.dumps(template['initial']))
  names=('left','right')
  init['robot_base_world']=dict(zip(names,receipt['initialization']['roots']))
  init['objects_task']['block']=b
  init['objects_task']['target']=receipt['initialization']['target_pose']
  spec=dict(config=template['config'],initial=init,
   at=dict(actual_arm=q[:,:6].tolist(),actual_fingers=q[:,6:].tolist(),
           target_fingers=np.clip(start_f,0,.044).tolist(),
           contact_state=dict(phase='approach',holders=[],block_pose_task=pose_from_wxyz(b[:3],b[3:]).tolist())),
   request=dict(stamp=dict(episode=f'native-{ep}',tick=start,sim_s=start*.004),state=template['request']['state']),
   source=dict(episode=episode,seed=receipt['input']['seed'],controls_sha256=receipt['input']['controls_sha256'],
               native_trace_sha256=receipt['physics_trace_sha256'],native_gripper_scale=[-.01,.045]),
   reference=nominal)
  (out/f'{ep}.json').write_text(json.dumps(spec,indent=2)+'\n')
# GPU job's actual first contact, reconstruct from directly preceding physical tick.
job='127999421'
outer=args.gpu_archive/job
d=outer/'trials/unified-seed1-arx-full-w4/episode'
manifest=json.loads((outer/'archive-manifest.json').read_text())
archive_config=outer/'release/configs/unified-seed1-arx-full-w4.json'
if json.loads(archive_config.read_text()) != template['config']:
 raise ValueError('archived GPU configuration differs from the CPU snapshot template')
for name in ('events.jsonl','action-00001.npy','initial-readback.json'):
 p=d/name
 if sha256(p.read_bytes()).hexdigest()!=manifest[str(p.relative_to(outer))]['sha256']:raise ValueError(p)
q=0; begin=None; pre=None; row=None
for line,s in enumerate((d/'events.jsonl').open(),1):
 x=json.loads(s)
 if x['kind']=='policy_request':
  q+=1
  if q==1:request=x
 if x['kind']=='approved_plan' and x['row_in_prefix']==36 and q==1:
  begin=pre
  plan=x['receipt']
  row=36
 if x.get('physics_step_count') and x.get('contact_state'):
  if x['contact_state']['contact_fingers']['left']:
   assert q==1 and row==36 and begin['tick']<x['tick']
   break
  pre=x
actions=np.load(d/'action-00001.npy')
spec=dict(config=template['config'],initial=json.loads((d/'initial-readback.json').read_text()),
          at=begin,request=dict(stamp=dict(request['stamp'],tick=begin['tick'],sim_s=begin['sim_s']),state=request['state']),
          reference=dict(first_contact_line=line,physics_at_first_contact=x,approved=plan,original_request_tick=request['stamp']['tick']),
          job=job,query=1,row=row,chunk=actions.tolist(),
          action_sha256=sha256((d/'action-00001.npy').read_bytes()).hexdigest(),
          events_sha256=sha256((d/'events.jsonl').read_bytes()).hexdigest())
(out/'first-push.json').write_text(json.dumps(spec,indent=2)+'\n')
print('prepared',out,'GPU',begin['tick'],x['tick'],line)
