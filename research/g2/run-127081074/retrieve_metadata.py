from pathlib import Path
import json,base64,hashlib,io,zipfile,subprocess
p=Path('results/pace-g2e-127081074')
mem=io.BytesIO()
inventory=[]
with zipfile.ZipFile(mem,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(p.rglob('*')):
  if not f.is_file():continue
  rel=str(f.relative_to(p));size=f.stat().st_size
  selected=(f.suffix in ('.json','.tsv','.txt','.log','.csv') and size<2000000) or (f.name=='phases.jsonl')
  row=dict(path=rel,bytes=size,retrieved=selected)
  if selected:
   data=f.read_bytes();row['sha256']=hashlib.sha256(data).hexdigest();z.writestr(rel,data)
  if f.name=='events.jsonl':
   lines=f.read_text().splitlines()
   z.writestr(rel+'.tail60', '\n'.join(lines[-60:])+'\n')
   row['event_lines']=len(lines)
  inventory.append(row)
 z.writestr('remote-file-inventory.json',json.dumps(inventory,indent=2))
 z.writestr('sacct.txt',subprocess.check_output(['sacct','-j','127081074','--format=JobID,State,ExitCode,Elapsed,Start,End,AllocTRES%80','-P']))
 z.writestr('slurm-output.log',Path('logs/pace-g2e-127081074.out').read_bytes())
data=mem.getvalue()
print('G2JOB_ARCHIVE_BEGIN',len(data),hashlib.sha256(data).hexdigest())
s=base64.b64encode(data).decode()
for i in range(0,len(s),120): print(s[i:i+120])
print('G2JOB_ARCHIVE_END')

