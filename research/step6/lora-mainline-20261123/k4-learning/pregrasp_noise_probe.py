"""Fixed real expert observations, multiple fixed noise keys; no action selection."""
import json
from pathlib import Path
import sys
import numpy as np
from openpi_client import msgpack_numpy
from scripts.step6_arx.protocol import socket_for,decode,actions_from

root=Path(sys.argv[1]); out=Path(sys.argv[2]);out.mkdir()
cases=root/'tools/shandong/cases'
manifest=json.loads((cases/'manifest.json').read_text())
results=[]
with socket_for('ws://127.0.0.1:51923',10) as ws:
    (out/'metadata.json').write_text(json.dumps(decode(ws.recv(timeout=30)),indent=2)+'\n')
    for item in manifest['entries']:
        if item['phase']!='pregrasp':continue
        with np.load(cases/item['file']) as case:
            for query in (2,3,4):
                request=dict(state=case['state'],images={k:case[k] for k in ('head_camera','left_camera','right_camera')},
                             prompt=case['prompt'].item(),query_index=query)
                ws.send(msgpack_numpy.packb(request));pred=actions_from(decode(ws.recv(timeout=120)))
                np.save(out/f'{item["name"]}-q{query}.npy',pred)
                hits=np.flatnonzero(pred[:,6]<=.9)
                record=dict(name=item['name'],query=query,first_close=int(hits[0]) if len(hits) else None,
                            grip_k4=pred[:4,6].tolist(),joint_delta_k4=(pred[:4,:6]-case['state'][:6]).tolist())
                results.append(record)
                (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
                print(json.dumps(record),flush=True)
(out/'complete.json').write_text(json.dumps(dict(count=len(results),physical_execution=False,
    action_selection=False,fixed_observation=True))+'\n')
