"""A non-destructive action-label view over consecutive identical observations.

Only exact equality of drive state and all three decoded RGB images permits a
merge. Time is observation-event order, not an inferred physical interval.
Inputs, original sampling indices, splits and normalization remain unchanged.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from PIL import Image

CAMERAS=('head_camera','left_camera','right_camera')


def rgb(blob):
    image=Image.open(io.BytesIO(blob)); result=np.asarray(image.convert('RGB'))
    return result if image.info.get('comment')==b'XPL-RGB1' else result[...,::-1]


def build(path,output):
    result=dict(schema='consecutive_exact_observation_events_v1',physical_frame_times='unknown',splits={})
    with h5py.File(path,'r') as f:
        for split in ('train','val','test'):
            result['splits'][split]={}
            for episode in sorted(f[split],key=int):
                g=f[split][episode];q=g['drive_target'][:];keep=[];raw_to_event=[]
                previous_blobs=None
                for t in range(len(q)):
                    same=False
                    if t and np.array_equal(q[t],q[t-1]):
                        old=previous_blobs or [bytes(g[c][t-1]) for c in CAMERAS]
                        new=[bytes(g[c][t]) for c in CAMERAS]
                        same=all(a==b or np.array_equal(rgb(a),rgb(b)) for a,b in zip(old,new))
                        previous_blobs=new
                    else:previous_blobs=None
                    if same:keep[-1]=t
                    else:keep.append(t)
                    raw_to_event.append(len(keep)-1)
                row=dict(raw_frames=len(q),unique_frames=len(keep),representatives=keep,raw_to_event=raw_to_event)
                result['splits'][split][str(int(episode))]=row
                print(json.dumps(dict(split=split,episode=int(episode),raw=len(q),unique=len(keep))),flush=True)
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    result['source_sha256']=h.hexdigest()
    output.write_text(json.dumps(result,indent=2)+'\n')


def relabel(dataset,mapping,split,episode,t):
    sample=dataset.read(split,episode,t,terminal=True)
    row=mapping['splits'][split][str(episode)]
    event=row['raw_to_event'][t]
    indices=row['representatives'][event+1:event+51]
    # At an already-terminal duplicate, the original archive still has a hold
    # command at its last row. Retain that real command as the valid target.
    if not indices:indices=[row['raw_frames']-1]
    valid=len(indices)
    padded=indices+[indices[-1]]*(50-valid)
    q=dataset.groups[split][episode]['drive_target'][:]
    sample['actions']=q[padded]
    sample['valid_rows']=valid
    sample['target_raw_indices']=padded
    return sample


if __name__=='__main__':
    build(Path(sys.argv[1]),Path(sys.argv[2]))
