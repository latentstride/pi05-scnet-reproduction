"""Same five untouched Val pregrasp observations, four predetermined noise keys."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
from scripts.step5_arx.data import Dataset

root,out=map(Path,sys.argv[1:]);out.mkdir()
entries=[]
with Dataset(root/'data/step5-arx/arx-drive-target-v1.h5') as d:
    for ep,g in sorted(d.groups['val'].items()):
        close=int(np.flatnonzero(np.diff(g['drive_target'][:,6]) < -1e-4)[0])+1
        row=close-1;sample=d.read('val',ep,row)
        name=f'val-ep{ep:02d}-pregrasp';p=out/f'{name}.npz'
        np.savez_compressed(p,state=sample['state'],expert=sample['actions'],prompt=sample['prompt'],**sample['images'])
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        for query in (1,2,3,4):
            entries.append(dict(name=f'{name}-q{query}',file=p.name,sha256=sha,episode=ep,
                                phase='pregrasp',observation_frame=row,query_index=query))
(out/'manifest.json').write_text(json.dumps(dict(split='val',entries=entries,
    query_indices=[1,2,3,4],selection='all five Val episodes; no favorable noise selection'),indent=2)+'\n')
print(json.dumps(dict(count=len(entries),episodes=5)),flush=True)
