"""Exact input aliasing near closure; do not invent missing physical timestamps."""
import hashlib
import io
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from PIL import Image

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir()
records=[]
with h5py.File(root/'data/step5-arx/arx-drive-target-v1.h5','r') as f:
    for split in ('train','val','test'):
        for ep in sorted(f[split],key=int):
            group=f[split][ep];q=group['drive_target'][:]
            close=int(np.flatnonzero(np.diff(q[:,6]) < -1e-4)[0])+1
            rows=[];hashes={}
            for t in range(close-5,close+1):
                h=hashlib.sha256(q[t].tobytes())
                for camera in ('head_camera','left_camera','right_camera'):
                    im=Image.open(io.BytesIO(bytes(group[camera][t])))
                    rgb=np.asarray(im.convert('RGB'))
                    if im.info.get('comment')!=b'XPL-RGB1':rgb=rgb[...,::-1]
                    h.update(rgb.tobytes())
                key=h.hexdigest();hashes.setdefault(key,[]).append(t)
                hits=np.flatnonzero(q[t+1:t+51,6]<=.9)
                rows.append(dict(row=t,input_sha256=key,first_close=int(hits[0]) if len(hits) else None,
                                 next_grip=q[t+1:t+5,6].tolist()))
            conflicts=[]
            for key,indices in hashes.items():
                if len(indices)<2:continue
                labels=np.stack([q[t+1:t+5,6] for t in indices])
                if np.array_equal(labels,np.broadcast_to(labels[0],labels.shape)):continue
                conflicts.append(dict(rows=indices,input_sha256=key,
                    grip_labels=labels.tolist(),minimum_equal_weight_k4_grip_mse=float(np.var(labels,axis=0).mean())))
            records.append(dict(split=split,episode=int(ep),close_frame=close,
                                rows=rows,exact_input_conflicts=conflicts,physical_timestamps='unknown'))
summary={split:dict(episodes=sum(r['split']==split for r in records),
    episodes_with_exact_input_conflicts=sum(r['split']==split and bool(r['exact_input_conflicts']) for r in records),
    conflicting_groups=sum(len(r['exact_input_conflicts']) for r in records if r['split']==split))
    for split in ('train','val','test')}
(out/'labels.json').write_text(json.dumps(records,indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
