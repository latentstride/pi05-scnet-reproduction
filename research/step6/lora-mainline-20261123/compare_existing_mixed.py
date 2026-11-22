"""Evaluate the already loaded mixed LoRA on the existing Test anchor inputs."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from openpi_client import msgpack_numpy
from scripts.step6_arx.protocol import socket_for,decode,actions_from,record_wire
from scripts.step5_arx.data import CAMERAS
from g2_assets import load_arms
from g2_kinematics import pose_error

p=argparse.ArgumentParser()
p.add_argument('root',type=Path);p.add_argument('output',type=Path)
p.add_argument('--cases',type=Path)
p.add_argument('--port',type=int,default=51899)
p.add_argument('--identity',default='e0fd7406d5797999f167affd1ef6a26e5ed7167d9981fb0928e11b9fa87fd0b6')
a=p.parse_args();a.output.mkdir()
cases=a.cases or a.root/'tools/shandong/cases'
manifest=json.loads((cases/'manifest.json').read_text())
bundle=a.root/'tools/a800-closed-loop'
safety=json.loads((bundle/'suite/docs/step6/arx_safety_profile.json').read_text())
arms=load_arms(bundle/'PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/assets',
    bundle/'pr5/research/g2/g2e-20261106-r1/release/asset-audit.json','ARX-X5',safety['calibration'])
results=[]
with socket_for(f'ws://127.0.0.1:{a.port}',10) as ws:
    metadata=decode(ws.recv(timeout=30))
    if metadata['step6_identity']['training_identity_sha256']!=a.identity:
        raise ValueError('warm server is not the existing mixed step-0400')
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for e in manifest['entries']:
        path=cases/e['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=e['sha256']:raise ValueError('anchor changed')
        with np.load(path) as c:
            request=dict(state=c['state'],images={cam:c[cam] for cam in CAMERAS},
                         prompt=c['prompt'].item(),query_index=e['query_index'])
            directory=a.output/e['name'];directory.mkdir()
            packed=msgpack_numpy.packb(request);wire_in=record_wire(directory,e['query_index'],'request',packed)
            ws.send(packed);reply=ws.recv(timeout=120)
            wire_out=record_wire(directory,e['query_index'],'response',reply)
            pred=actions_from(decode(reply));np.save(directory/'prediction.npy',pred)
            label=c['expert'];bases=c['bases'] if 'bases' in c else np.repeat(np.eye(4)[None],2,axis=0)
            def tcp(q):
                return np.array([[base@arm.chain.fk(row[i*7:i*7+6])@arm.offset()
                    for i,(base,arm) in enumerate(zip(bases,arms))] for row in q])
            expected=tcp(label);actual=tcp(pred)
            err=np.array([[pose_error(x,y) for x,y in zip(er,pr)] for er,pr in zip(expected,actual)])
            def close(g,threshold):
                hits=np.flatnonzero(g<=threshold);return int(hits[0]) if len(hits) else None
            record=dict(name=e['name'],query=e['query_index'],request=wire_in,response=wire_out,
                left_k4_tcp_mm=(1000*np.linalg.norm(err[:4,0,:3],axis=1)).tolist(),
                left_k4_tcp_deg=np.degrees(np.linalg.norm(err[:4,0,3:],axis=1)).tolist(),
                left_k4_joint_mae_rad=float(np.mean(abs(pred[:4,:6]-label[:4,:6]))),
                left_grip_k4=pred[:4,6].tolist(),expert_grip_k4=label[:4,6].tolist(),
                first_close_model={str(t):close(pred[:,6],t) for t in (.9,.5,.1)},
                first_close_expert={str(t):close(label[:,6],t) for t in (.9,.5,.1)},
                h50_left_tcp_mm=(1000*np.linalg.norm(err[:,0,:3],axis=1)).tolist())
            results.append(record)
            (a.output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:record[k] for k in ('name','left_k4_tcp_mm','left_grip_k4')}),flush=True)
(a.output/'complete.json').write_text(json.dumps(dict(count=len(results),checkpoint=metadata,
    source_cases=str(cases),decode_boundary='post-decoding 50x14; no controller action changes',
    physical_execution=False),indent=2)+'\n')
