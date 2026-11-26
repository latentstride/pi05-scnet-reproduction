import json,h5py
from scripts.step5_arx.data import Dataset as OriginalDataset,SCHEMA,PROMPT,partitions
class Dataset(OriginalDataset):
 def __init__(self,path):
  self.file=h5py.File(path,'r');self.norm_stats=json.loads(self.file.attrs['norm_stats']);self.split=json.loads(self.file.attrs['split_manifest'])
  assert self.file.attrs['schema']==SCHEMA and self.file.attrs['prompt']==PROMPT
  assert self.file.attrs['dataset_variant']=='150newclean-train-only-newnorm-v1'
  orig=partitions();assert self.split==dict(train=list(range(50,200)),val=orig['val'],test=orig['test'])
  self.groups={s:{int(k):g for k,g in self.file[s].items()} for s in ('train','val','test')}
  assert {s:sorted(g) for s,g in self.groups.items()}==self.split
