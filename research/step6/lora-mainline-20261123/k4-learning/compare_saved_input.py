"""Fixed saved input/noise comparison; no physical actions or state reconstruction."""
import json,sys
from pathlib import Path
import numpy as np
from openpi_client import msgpack_numpy
from scripts.step6_arx.protocol import socket_for,decode,actions_from,record_wire
from g2_assets import load_arms
from g2_kinematics import pose_error
root,trial=map(Path,sys.argv[1:3])
records=[json.loads(x) for x in (trial/'trajectory.jsonl').read_text().splitlines()]
query=max(r['query_index'] for r in records if r['kind']=='policy_request')
physical=None
for r in records:
 if r['kind']=='physics':physical=r
 if r['kind']=='policy_request' and r['query_index']==query:
  request=decode((trial/r['payload']['file']).read_bytes()); break
response=next(r for r in records if r['kind']=='policy_response' and r['query_index']==query)
original=actions_from(decode((trial/response['payload']['file']).read_bytes()))
out=trial/'saved-input-comparison';out.mkdir()
bundle=root/'tools/a800-closed-loop';safety=json.loads((bundle/'suite/docs/step6/arx_safety_profile.json').read_text())
arms=load_arms(bundle/'PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/assets',bundle/'pr5/research/g2/g2e-20261106-r1/release/asset-audit.json','ARX-X5',safety['calibration'])
def poses(pred):
 return np.asarray([[arm.chain.fk(row[i*7:i*7+6])@arm.offset() for i,arm in enumerate(arms)] for row in pred])
reference=poses(original)
results=[]
for name,port,identity,qpos in [
 ('raw_labels_same_input',51923,'591dc20b4af2a4174f19c8eaa1931e914b8b8f28b561540f0d674c5e85e67c3d',False),
 ('repaired_actual_arm_qpos',51924,'5e0f1347e6a18f2f297c691ea0b110470e885813fafcc48b96577c61131d5b6b',True)]:
 req=dict(request);req['state']=np.asarray(request['state']).copy()
 if qpos:
  for side in range(2):req['state'][7*side:7*side+6]=np.asarray(physical['actual_qpos'])[side]
 directory=out/name;directory.mkdir()
 with socket_for(f'ws://127.0.0.1:{port}',20) as ws:
  metadata=decode(ws.recv(timeout=30))
  assert metadata['step6_identity']['training_identity_sha256']==identity
  packed=msgpack_numpy.packb(req);sent=record_wire(directory,query,'request',packed)
  ws.send(packed);reply=ws.recv(timeout=120);received=record_wire(directory,query,'response',reply)
 pred=actions_from(decode(reply));np.save(directory/'prediction.npy',pred)
 changes=np.asarray([[pose_error(x,y) for x,y in zip(ar,br)] for ar,br in zip(reference,poses(pred))])
 def first(g,t):
  h=np.flatnonzero(g<=t);return int(h[0]) if len(h) else None
 result=dict(name=name,query=query,metadata=metadata,request=sent,response=received,
  left_grip_h50=pred[:,6].tolist(),first_close={str(t):first(pred[:,6],t) for t in (.9,.5,.1)},
  changed_from_repaired_original_k4_tcp_mm=(1000*np.linalg.norm(changes[:4,0,:3],axis=-1)).tolist(),
  changed_from_repaired_original_k4_tcp_deg=np.degrees(np.linalg.norm(changes[:4,0,3:],axis=-1)).tolist(),
  changed_state=(np.asarray(req['state'])-np.asarray(request['state'])).tolist())
 results.append(result)
 (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(result),flush=True)

