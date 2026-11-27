"""Verify native warmup force restoration, then reuse that live reset for LoRA execution."""
import importlib.util,json,sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from scripts.step6_arx import evaluate
from scripts.step6_arx.protocol import socket_for,decode,actions_from,record_wire
from scripts.step5_arx.data import Dataset
from openpi_client import msgpack_numpy
from g2_episode import json_value
from g2_kinematics import pose_error
from physical_checkpoint import save_checkpoint

root,out,oldtrial=map(Path,sys.argv[1:4]);out.mkdir()
sim=root/'tools/a800-closed-loop'
spec=importlib.util.spec_from_file_location('scripts.step6_arx.scene',Path(__file__).with_name('pure_scene.py'))
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
evaluate.install_safety(sim/'pr5')
plan=json.loads((sim/'suite/docs/step6/pairs.json').read_text())
safety=json.loads((sim/'suite/docs/step6/arx_safety_profile.json').read_text())
args=NS(pr5_root=sim/'pr5',pace_root=sim/'PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1',
 asset_audit=sim/'pr5/research/g2/g2e-20261106-r1/release/asset-audit.json',
 cpu_fixture=False,gpu_short=False,headless=False,
 expected_training_identity='5e0f1347e6a18f2f297c691ea0b110470e885813fafcc48b96577c61131d5b6b')
old=json.loads((oldtrial/'receipt.json').read_text())
seed=old['scene_seed'];episode=f'seed-{seed:02d}-lora'
sceneout=out/'scene';sceneout.mkdir()
def write(name,obj):
 (out/name).write_text(json.dumps(obj,indent=2,default=json_value)+'\n')
with native.open_scene(args,plan,safety,seed,sceneout,episode) as (scene,setup):
 initial=scene.reset(dict(episode_id=episode));extras=scene.initial_extras()
 write('corrected-initial.json',dict(physics=initial,readback=extras))
 request=scene.request(0)
 old_request_event=next(json.loads(x) for x in (oldtrial/'trajectory.jsonl').read_text().splitlines() if json.loads(x)['kind']=='policy_request')
 old_reply_event=next(json.loads(x) for x in (oldtrial/'trajectory.jsonl').read_text().splitlines() if json.loads(x)['kind']=='policy_response')
 oldrequest=decode((oldtrial/old_request_event['payload']['file']).read_bytes())
 oldpred=actions_from(decode((oldtrial/old_reply_event['payload']['file']).read_bytes()))
 with Dataset(root/'data/step5-arx/arx-drive-target-v1.h5') as data:
  sample=data.read('train',seed,0,terminal=True)
 expert_request=dict(state=sample['state'],images=sample['images'],prompt=sample['prompt'],query_index=0)
 predictions={'old_live_initial':oldpred}
 metadata=None
 for name,req in [('corrected_live_initial',request),('same_seed_expert_initial',expert_request)]:
  directory=out/name;directory.mkdir()
  with socket_for('ws://127.0.0.1:51924',20) as socket:
   metadata=decode(socket.recv(timeout=30))
   assert metadata['step6_identity']['training_identity_sha256']==args.expected_training_identity
   packed=msgpack_numpy.packb(req);record_wire(directory,0,'request',packed)
   socket.send(packed);reply=socket.recv(timeout=180);record_wire(directory,0,'response',reply)
   predictions[name]=actions_from(decode(reply));np.save(directory/'prediction.npy',predictions[name])
 def tcp(q):
  return np.asarray([[arm.chain.fk(row[i*7:i*7+6])@arm.offset() for i,arm in enumerate(scene.arms)] for row in q])
 expected=tcp(sample['actions'])
 report=dict(seed=seed,change='restore native passive force calls during gripper initialization only',
  old_actual_qpos=old['initial']['readback']['qpos'],fixed_actual_qpos=extras['qpos'],
  command_unchanged=bool(np.array_equal(extras['command'],old['initial']['readback']['command'])),
  source_state_matches=bool(np.array_equal(request['state'],sample['state'])),
  physical_restore_used=False,live_reset_reused_for_rollout=True,metrics={})
 for name,pred in predictions.items():
  err=np.asarray([pose_error(a,b) for a,b in zip(expected[:4,0],tcp(pred)[:4,0])])
  h=np.flatnonzero(pred[:,6]<=.9)
  report['metrics'][name]=dict(k4_tcp_mm=(1000*np.linalg.norm(err[:,:3],axis=-1)).tolist(),
    k4_tcp_deg=np.degrees(np.linalg.norm(err[:,3:],axis=-1)).tolist(),
    first_close=int(h[0]) if len(h) else None,left_grip_k4=pred[:4,6].tolist())
 report['images_mean_abs_difference']={cam:dict(
  fixed_vs_expert=float(np.mean(abs(np.asarray(request['images'][cam],float)-np.asarray(expert_request['images'][cam],float)))),
  old_vs_expert=float(np.mean(abs(np.asarray(oldrequest['images'][cam],float)-np.asarray(expert_request['images'][cam],float))))) for cam in request['images']}
 state=np.asarray(extras['command']).reshape(2,7);actual=np.asarray(extras['qpos'])[:,:6]
 report['fixed_drive_actual_tcp']=[]
 for i,arm in enumerate(scene.arms):
  e=pose_error(arm.chain.fk(actual[i])@arm.offset(),arm.chain.fk(state[i,:6])@arm.offset())
  report['fixed_drive_actual_tcp'].append(dict(mm=float(1000*np.linalg.norm(e[:3])),deg=float(np.degrees(np.linalg.norm(e[3:])))))
 write('initial-counterfactual.json',report);print(json.dumps(report),flush=True)
 # Continue in this exact live scene. No qpos/actor reset or expert actions.
 adapter=NS(seed=None,last_stamp=None);coordinator=NS()
 original_request=scene.request;original_execute=scene.execute_native_row
 def saved_request(query):
  req=original_request(query)
  save_checkpoint(out/'latest-request-state.pkl',scene,adapter,coordinator,dict(query=query,request=req,policy_seed=42))
  return req
 def saved_execute(row,remaining,log):
  try:return original_execute(row,remaining,log)
  except Exception:
   save_checkpoint(out/'failure-state.pkl',scene,adapter,coordinator,dict(target=row,policy_seed=42))
   raise
 scene.request=saved_request;scene.execute_native_row=saved_execute
 original_reset=scene.reset
 def same_live_reset(spec):
  if spec['episode_id']!=episode or scene.tick!=0:raise RuntimeError('not the untouched live reset')
  return initial
 scene.reset=same_live_reset
 @contextmanager
 def reuse(*unused,**kwargs):
  yield scene,setup
 native.open_scene=reuse
 reference=dict(physics=initial,readback=extras)
 receipt=evaluate.trial(args,plan,safety,seed,'lora',out/'rollout','ws://127.0.0.1:51924',
  reference,old['checkpoint'])
 receipt['corrected_reset_self_consistency']=receipt['pair_match']
 receipt['pair_match']=None
 receipt['initial_reference_kind']='corrected_native_reset; not paired with historical drooped reset'
 receipt['original_initial']=True
 receipt['video']='../scene/head.mp4'
 receipt['initial_counterfactual']='../initial-counterfactual.json'
 (out/'rollout/receipt.json').write_text(json.dumps(receipt,indent=2,default=json_value)+'\n')
 write('complete.json',dict(outcome=receipt['outcome'],task_success=receipt['task_success'],
  failure=receipt['failure'],actions=receipt['policy_actions'],analysis=receipt['analysis']))
 print(json.dumps(dict(outcome=receipt['outcome'],actions=receipt['policy_actions'],failure=receipt['failure'])),flush=True)

