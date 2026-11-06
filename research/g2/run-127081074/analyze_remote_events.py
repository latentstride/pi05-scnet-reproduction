from pathlib import Path
from collections import Counter
import json,base64,hashlib,io,zipfile,math
p=Path('results/pace-g2e-127081074');mem=io.BytesIO();summaries={}
with zipfile.ZipFile(mem,'w',zipfile.ZIP_DEFLATED) as z:
 for d in sorted((p/'trials').iterdir()):
  if not d.is_dir():continue
  requests=[];rejections=[];first_closed={};states=Counter();first_hold={};tick=0;nsteps=0;sumq=[0.0]*12;maxq=[0.0]*12;sump=[0.0,0.0];maxp=[0.0,0.0];maxangle=[0.0,0.0];lastguard=None;lastcontact=None;laststep=None;lastrequest=None;contacts_at_first_hold={};state_at_first_hold={}
  with (d/'events.jsonl').open() as stream:
   for line in stream:
    r=json.loads(line);kind=r.get('kind')
    if kind=='native_action_request':
     requests.append(r);lastrequest=r
     for side,col in [('left',6),('right',13)]:
      if r['normalized_qpos_request'][col]<0.5 and side not in first_closed:first_closed[side]=r
    if kind=='native_contact_sample':lastcontact=r
    if kind=='native_guard_state':
     lastguard=r;holders=tuple(sorted(r['contact_state']['holders']));states[str(holders)]+=1
     if holders and str(holders) not in first_hold:
      first_hold[str(holders)]=r['tick'];contacts_at_first_hold[str(holders)]=lastcontact;state_at_first_hold[str(holders)]=r
    if kind=='native_guard_rejection':rejections.append(dict(rejection=r,last_request=lastrequest,last_guard=lastguard,last_contact=lastcontact,last_step=laststep))
    if r.get('physics_step_count')==1:
     nsteps+=1;tick=max(tick,r['tick']);laststep=r
     if 'actual_arm' in r and 'target_arm' in r:
      errors=[a-b for av,bv in zip(r['target_arm'],r['actual_arm']) for a,b in zip(av,bv)]
      for i,e in enumerate(errors):sumq[i]+=e*e;maxq[i]=max(maxq[i],abs(e))
      for i,e in enumerate(r['position_error_m']):sump[i]+=e*e;maxp[i]=max(maxp[i],e)
      for i,e in enumerate(r['orientation_error_deg']):maxangle[i]=max(maxangle[i],e)
  summaries[d.name]=dict(physics_steps=nsteps,final_tick=tick,request_count=len(requests),first_closed_requests=first_closed,holder_samples=dict(states),first_holder_ticks=first_hold,arm_joint_rmse_rad=[math.sqrt(s/nsteps) for s in sumq],arm_joint_max_abs_rad=maxq,tcp_position_rmse_m=[math.sqrt(s/nsteps) for s in sump],tcp_position_max_m=maxp,orientation_max_deg=maxangle,last_guard=lastguard)
  z.writestr('derived/'+d.name+'/requests.json',json.dumps(requests))
  z.writestr('derived/'+d.name+'/rejection-context.json',json.dumps(rejections))
  z.writestr('derived/'+d.name+'/first-holder-context.json',json.dumps(dict(states=state_at_first_hold,contacts=contacts_at_first_hold)))
  for pattern in ('plan-*.npy','action-*.npy'):
   for f in sorted(d.glob(pattern)):z.writestr(str(f.relative_to(p)),f.read_bytes())
 z.writestr('derived/event-summary.json',json.dumps(summaries,indent=2))
data=mem.getvalue();print('G2JOB_ARCHIVE_BEGIN',len(data),hashlib.sha256(data).hexdigest())
s=base64.b64encode(data).decode()
for i in range(0,len(s),120): print(s[i:i+120])
print('G2JOB_ARCHIVE_END')

