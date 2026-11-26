import json,sys,hashlib,os
from pathlib import Path
import h5py,numpy as np
from scripts.step5_arx.data import stats,delta,partitions,CAMERAS,digest
root=Path(sys.argv[1]);out=Path(sys.argv[2]);source=root/'data/step5-arx-190clean-v1/arx-drive-target-190clean-v1.h5'
dest=root/'data/step5-arx-clean150-newnorm-20261126-v1.h5'
if dest.exists():raise FileExistsError(dest)
manifest=json.loads((root/'results/step5-arx-clean190-auto-20261125-v1/collection-manifest.json').read_text())
assert len(manifest['accepted'])==150
states=[];actions=[];original=partitions();splits=dict(train=list(range(50,200)),val=original['val'],test=original['test'])
before=dict(size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns)
with h5py.File(source,'r') as old,h5py.File(dest.with_suffix('.partial'),'x') as new:
 for k,v in old.attrs.items():new.attrs[k]=v
 for split,eps in splits.items():
  new.create_group(split)
  for ep in eps:
   old.copy(old[f'{split}/{ep}'],new[split],name=str(ep))
   if split=='train':
    q=old[f'train/{ep}/drive_target'][:];s=q[:-1]
    future=np.stack([np.pad(q[t+1:t+51],((0,max(0,t+51-len(q))),(0,0)),mode='edge') for t in range(len(s))])
    states.append(s);actions.append(delta(s,future).reshape(-1,14))
 norm={name:stats(np.concatenate(values)) for name,values in [('state',states),('actions',actions)]}
 new.attrs['norm_stats']=json.dumps(norm,sort_keys=True)
 new.attrs['dataset_variant']='150newclean-train-only-newnorm-v1';new.attrs['split_manifest']=json.dumps(splits)
 new.attrs['composition']=json.dumps(dict(train_new_clean=150,train_old_clean=0,val_old_clean=5,test_old_clean=5))
os.link(dest.with_suffix('.partial'),dest);dest.with_suffix('.partial').unlink()
assert before==dict(size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns)
obj=dict(dataset=str(dest),sha256=digest(dest),splits=splits,norm_stats=norm,train_seeds=[r['seed'] for r in manifest['accepted']],
 stats_source='150 Train episodes only; every training observation with H50 next-saved-row/terminal-edge targets; raw unperturbed state and arm-delta targets; matches Dataset terminal=True',
 capture='same demo_clean/save_freq15 collection configuration; physical intervals not assumed constant',original_source_unchanged=True)
(out/'dataset150.json').write_text(json.dumps(obj,indent=2)+'\n');(out/'norm_stats.json').write_text(json.dumps(norm,indent=2)+'\n')
print('DATA150_READY',str(dest),flush=True)
