"""CPU-only resumable HDF5 audit; nothing is dropped or imputed."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,hashlib,io,csv,time,traceback
import h5py,numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1]
def analyze(p):
 robot=p.parent.name;episode=int(p.stem.replace('episode',''));key=f'{robot}-episode{episode}';sha=hashlib.sha256(p.read_bytes()).hexdigest();dest=R/'data/episodes'/f'{key}.json';dest.parent.mkdir(exist_ok=True)
 if dest.exists():
  old=json.loads(dest.read_text())
  if old.get('sha256')==sha and old.get('pipeline_version')=='v2':return old
 rec=dict(robot=robot,episode=episode,file=f'raw/{robot}/{p.name}',sha256=sha,bytes=p.stat().st_size,status='read_failed',retained=True,pipeline_version='v2')
 try:
  fields=[];anomalies=[];allhash=[];bad_rgb=0;images=0;empty=0
  with h5py.File(p) as h:
   required=['joint_action/vector','joint_action/left_arm','joint_action/right_arm','joint_action/left_gripper','joint_action/right_gripper','endpose/left_endpose','endpose/right_endpose']
   required += ['endpose/left_gripper','endpose/right_gripper']+[f'observation/{camera}/{field}' for camera in ['head_camera','left_camera','right_camera'] for field in ['rgb','intrinsic_cv','extrinsic_cv','cam2world_gl']]
   missing=[k for k in required if k not in h];rec['missing_fields']=missing
   if missing:raise ValueError('required fields missing: '+','.join(missing))
   a=h['joint_action/vector'][:];n=len(a);rec['frames']=n
   if a.shape!=(n,14) or n<1:raise ValueError('invalid state14 shape')
   fullhash=[hashlib.sha256() for _ in range(n)]
   def visit(k,v):
    nonlocal bad_rgb,images,empty
    if not isinstance(v,h5py.Dataset):return
    arr=v[:];isnum=arr.dtype.kind in 'fiu';bad=np.argwhere(~np.isfinite(arr)) if isnum else np.empty((0,1));has_rows=arr.ndim>0 and len(arr)==n
    finite=arr[np.isfinite(arr)] if isnum else np.array([])
    fields.append(dict(field=k,dtype=str(arr.dtype),shape=list(arr.shape),nonfinite=int(len(bad)),missing_or_empty=arr.size==0,min=float(finite.min()) if finite.size else None,max=float(finite.max()) if finite.size else None))
    if arr.size==0:empty+=1
    if arr.ndim and len(arr)!=n:anomalies.append(dict(type='length_mismatch',field=k,expected=n,actual=len(arr)))
    if len(bad):anomalies.append(dict(type='nonfinite',field=k,indices=bad.tolist()))
    if has_rows:
     for i in range(n):fullhash[i].update(k.encode()+np.asarray(arr[i]).tobytes())
    if k.endswith('/rgb') or k=='third_view_rgb':
     for i,b in enumerate(arr):
      images+=1
      try:
       im=Image.open(io.BytesIO(bytes(b)));im.load()
       if im.size!=(320,240) or im.mode!='RGB':anomalies.append(dict(type='image_shape_mode',field=k,frame=i,size=list(im.size),mode=im.mode))
      except Exception as e:bad_rgb+=1;anomalies.append(dict(type='image_decode',field=k,frame=i,error=type(e).__name__))
   h.visititems(visit)
   rec.update(fields=fields,anomalies=anomalies,numeric_nonfinite=sum(x['nonfinite'] for x in fields))
   if rec['numeric_nonfinite']:raise ValueError('nonfinite numeric values: raw retained; excluded from canonical and mapping')
   cat=np.column_stack([h['joint_action/left_arm'][:],h['joint_action/left_gripper'][:],h['joint_action/right_arm'][:],h['joint_action/right_gripper'][:]])
   diff=abs(cat-a);concat=float(diff.max());concat_indices=np.argwhere(diff!=0)
   if len(concat_indices):anomalies.append(dict(type='state_concat_mismatch',indices=concat_indices.tolist()))
   grips=a[:,[6,13]];gi=np.argwhere((grips<0)|(grips>1));
   if len(gi):anomalies.append(dict(type='gripper_out_of_range',indices=gi.tolist()))
   _,first,inv=np.unique(a,axis=0,return_index=True,return_inverse=True);dups=[int(i) for i in range(n) if first[inv[i]]!=i];adj=(np.where(np.all(a[1:]==a[:-1],axis=1))[0]+1).tolist()
   fh=[x.hexdigest() for x in fullhash];adjfull=[i for i in range(1,n) if fh[i]==fh[i-1]];full_duplicates=n-len(set(fh))
   normerrs=[];end=[]
   for arm in ['left','right']:
    ee=h[f'endpose/{arm}_endpose'][:];end.append(ee);err=abs(np.linalg.norm(ee[:,3:],axis=1)-1);normerrs.extend(err.tolist());idx=np.where(err>1e-6)[0].tolist()
    if idx:anomalies.append(dict(type='quaternion_norm',field=f'endpose/{arm}_endpose',indices=idx,tolerance=1e-6))
   timing=[k for k in h.attrs if k.lower() in ['timestamp','timestamps','dt','fps']]+[f['field'] for f in fields if any(w in f['field'].lower() for w in ['timestamp','/time','/fps'])]
   for side in ['left','right']:
    err=abs(h[f'joint_action/{side}_gripper'][:]-h[f'endpose/{side}_gripper'][:])
    if np.any(err):anomalies.append(dict(type='duplicate_gripper_mismatch',side=side,frames=np.where(err!=0)[0].tolist()))
   rec.update(status='read_ok',numeric_nonfinite=sum(x['nonfinite'] for x in fields),gripper_out_of_range=len(gi),quaternion_bad=sum(e>1e-6 for e in normerrs),quaternion_norm_max_error=max(normerrs),concat_max_error=concat,state_duplicate_rows=len(dups),consecutive_state_repeats=len(adj),full_frame_duplicates=full_duplicates,consecutive_full_frame_repeats=len(adjfull),state_duplicate_indices=dups,consecutive_state_indices=adj,consecutive_full_frame_indices=adjfull,empty_fields=empty,images=images,image_decode_errors=bad_rgb,timing_fields=timing,fields=fields,anomalies=anomalies,rows_removed=0,float32_max_abs_error=float(abs(a-a.astype('float32')).max()))
   np.savez_compressed(R/'data/episodes'/f'{key}.npz',state14=a.astype('float32'),left_endpose7=end[0],right_endpose7=end[1],frame_index=np.arange(n,dtype='int32'))
   preview={}
   for i in [0,(n-1)//2,n-1]:
    try:
     im=Image.open(io.BytesIO(bytes(h['observation/head_camera/rgb'][i])));name=f'{key}-frame{i}.jpg';im.save(R/'dashboard/previews'/name,quality=85);preview[str(i)]='previews/'+name
    except Exception:pass
   # Rounded display data only; numerical computations use NPZ/HDF5 full stored precision.
   t=dict(robot=robot,episode=episode,n=n,state14=np.round(a,6).tolist(),left_endpose7=np.round(end[0],7).tolist(),right_endpose7=np.round(end[1],7).tolist(),preview=preview,source_sha256=sha)
   (R/'data/episodes'/f'{key}-trajectory.json').write_text(json.dumps(t,separators=(',',':')))
 except Exception as e:rec.update(error_type=type(e).__name__,error=str(e),trace=traceback.format_exc())
 dest.write_text(json.dumps(rec,ensure_ascii=False,indent=2));print(key,rec['status'],rec.get('frames'),flush=True);return rec

def aggregate():
 recs=[json.loads(p.read_text()) for p in (R/'data/episodes').glob('*.json') if not p.name.endswith('-trajectory.json')];recs.sort(key=lambda r:(r['robot'],r['episode']))
 valid=[r for r in recs if r['status']=='read_ok'];flat=[]
 keys=['robot','episode','status','frames','bytes','sha256','numeric_nonfinite','gripper_out_of_range','quaternion_bad','quaternion_norm_max_error','concat_max_error','state_duplicate_rows','consecutive_state_repeats','full_frame_duplicates','consecutive_full_frame_repeats','empty_fields','images','image_decode_errors','rows_removed']
 for row in recs:flat.append({k:row.get(k) for k in keys})
 def writecsv(name,rows):
  if not rows:return
  with (R/'data'/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 writecsv('trajectory_quality.csv',flat);summary=[]
 for robot in ['aloha-agilex','arx-x5']:
  rr=[r for r in valid if r['robot']==robot];nn=[r['frames'] for r in rr]
  summary.append(dict(robot=robot,expected=50,read_ok=len(rr),read_failed=sum(r['status']!='read_ok' and r['robot']==robot for r in recs),not_processed=50-sum(r['robot']==robot for r in recs),frames=sum(nn),min_frames=min(nn) if nn else None,max_frames=max(nn) if nn else None,median_frames=float(np.median(nn)) if nn else None,**{k:sum(r[k] for r in rr) for k in ['images','image_decode_errors','numeric_nonfinite','gripper_out_of_range','quaternion_bad','state_duplicate_rows','consecutive_state_repeats','full_frame_duplicates','consecutive_full_frame_repeats','rows_removed']}))
 writecsv('embodiment_summary.csv',summary)
 inventory=[dict(robot=r['robot'],episode=r['episode'],**field) for r in valid for field in r['fields']];writecsv('field_inventory.csv',inventory)
 anomalies=[dict(robot=r['robot'],episode=r['episode'],**a) for r in recs for a in r.get('anomalies',[])];(R/'data/anomaly_index.json').write_text(json.dumps(anomalies,ensure_ascii=False,indent=2))
 result=dict(expected=100,processed=len(recs),read_ok=len(valid),total_frames=sum(r['frames'] for r in valid),unprocessed=[dict(robot=robot,episode=i) for robot in ['aloha-agilex','arx-x5'] for i in range(50) if not any(r['robot']==robot and r['episode']==i for r in recs)],robots=summary,anomaly_events=len(anomalies),source_revision='3dc3b798668feb99ac61cc9086d84cbcc3d79186',quality_parameters=dict(quaternion_norm_tolerance=1e-6,gripper_interval=[0,1],drop_policy='retain_all',timing='frame_index_not_physical_time',duplicate_definition='exact state14 equality within episode; full frame uses all stored datasets'))
 (R/'data/summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 trajectories=[json.loads((R/'data/episodes'/f"{r['robot']}-episode{r['episode']}-trajectory.json").read_text()) for r in valid]
 (R/'data/trajectories.json').write_text(json.dumps(trajectories,separators=(',',':')))
 with h5py.File(R/'data/unified.h5','w') as h:
  for r in valid:
   a=np.load(R/'data/episodes'/f"{r['robot']}-episode{r['episode']}.npz");g=h.create_group(f"{r['robot']}/episode{r['episode']}")
   for k in a:g.create_dataset(k,data=a[k],compression='gzip')
   g.attrs['source_sha256']=r['sha256'];g.attrs['original_timestamp']='missing';g.attrs['recorded_state_semantics']='recorded joint_action; not assumed measured qpos or independent control command'
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--seconds',type=int,default=1200);args=parser.parse_args();(R/'dashboard/previews').mkdir(exist_ok=True);(R/'data/episodes').mkdir(exist_ok=True);end=time.monotonic()+args.seconds
 while True:
  for p in sorted((R/'raw').glob('*/*.hdf5')):analyze(p)
  result=aggregate();print('COVERAGE',result['read_ok'],result['total_frames'],flush=True)
  if not args.watch or result['processed']==100 or time.monotonic()>end:break
  time.sleep(12)
