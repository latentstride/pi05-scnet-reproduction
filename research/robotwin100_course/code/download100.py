"""New resumable 100-member extractor; original sample script remains unchanged."""
from pathlib import Path
import concurrent.futures as cf,threading,importlib.util,json,hashlib,ssl,re,tempfile,zlib,os
R=Path(__file__).resolve().parents[1];REV='3dc3b798668feb99ac61cc9086d84cbcc3d79186'
spec=importlib.util.spec_from_file_location('original',R/'sources/sample_original.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def ctx():
 ca_bundle = os.environ.get('ROBOTWIN_CA_BUNDLE')
 return ssl.create_default_context(cafile=ca_bundle) if ca_bundle else ssl.create_default_context()
m.context=ctx
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--robots',nargs='+',default=['aloha-agilex','arx-x5']);parser.add_argument('--suffix',default='');args=parser.parse_args()
listing=json.loads((R/'sources/listing-huggingface.co.json').read_text());lock=threading.Lock();out=[]
def process(robot):
 item=next(x for x in listing if x['path'].endswith('/'+robot+'_clean_50.zip'));url=f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{REV}/'+item['path']
 remote=m.RangeFile(url,item['size'],item['size']*2)
 with m.zipfile.ZipFile(remote) as z:infos=[i for i in z.infolist() if i.filename.endswith(('.hdf5','.h5'))]
 infos.sort(key=lambda x:[int(n) if n.isdigit() else n for n in re.split(r'(\d+)',x.filename)])
 assert len(infos)==50
 (R/'sources'/f'{robot}-zip-index.json').write_text(json.dumps([dict(member=i.filename,bytes=i.file_size,compressed=i.compress_size,crc32=f'{i.CRC:08x}') for i in infos],indent=2))
 def one(info):
  dest=R/'raw'/robot/Path(info.filename).name;dest.parent.mkdir(exist_ok=True)
  rec={'robot':robot,'episode':int(re.search(r'episode(\d+)',dest.name).group(1)),'member':info.filename,'revision':REV,'archive_url':url,'archive_lfs_sha256_metadata':item['lfs']['oid'],'expected_bytes':info.file_size,'zip_crc32':f'{info.CRC:08x}'}
  try:
   cached=R.parent/'pi05/research/robotwin/data_probe'/robot/dest.name
   if not dest.exists() and cached.exists():
    import shutil;shutil.copy2(cached,dest)
   reused=dest.exists()
   if reused:
    data=dest.read_bytes()
    if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise ValueError('existing member CRC/length mismatch')
    received=0
   else:
    rf=m.RangeFile(url,item['size'],min(item['size'],100*1024*1024))
    with m.zipfile.ZipFile(rf) as zz:data=zz.read(info.filename)
    received=rf.transferred
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix='.partial', delete=False) as temporary:
     temporary.write(data)
    Path(temporary.name).replace(dest)
   rec.update(status='downloaded',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc_verified=True,reused=reused,http_bytes_received=received)
  except Exception as e:rec.update(status='download_failed',error_type=type(e).__name__,error=str(e)[:180])
  with lock:
   out.append(rec)
   with (R/('logs/download'+args.suffix+'.jsonl')).open('a') as f:f.write(json.dumps(rec)+'\n')
   (R/('data/download_manifest'+args.suffix+'.json')).write_text(json.dumps({'repository':'TianxingChen/RoboTwin2.0','revision':REV,'expected_trajectories':100,'full_archive_downloaded':False,'records':out},indent=2))
   print(robot,rec['episode'],rec['status'],len(out),flush=True)
 with cf.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(one,infos))
for robot in args.robots:process(robot)
print('COMPLETE',sum(x['status']=='downloaded' for x in out),'/100',flush=True)
