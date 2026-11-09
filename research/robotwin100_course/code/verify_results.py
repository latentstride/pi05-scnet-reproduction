from pathlib import Path
import json,csv,hashlib,zlib,sys,collections,time
import numpy as np,h5py
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
from map100 import setup
start=time.time();manifest=json.loads((R/'data/download_manifest.json').read_text());rows=list(csv.DictReader((R/'data/trajectory_quality.csv').open()));assert len(rows)==100
count=images=0;dups=0;seen=set()
with h5py.File(R/'data/unified.h5') as h:
 for r in rows:
  p=R/'raw'/r['robot']/f"episode{r['episode']}.hdf5";bs=p.read_bytes();rec=next(x for x in manifest['records'] if x['robot']==r['robot'] and str(x['episode'])==r['episode']);sha=hashlib.sha256(bs).hexdigest();assert sha==r['sha256']==rec['sha256'];assert f'{zlib.crc32(bs):08x}'==rec['zip_crc32'];seen.add(sha)
  with h5py.File(p) as raw:
   a=raw['joint_action/vector'][:];n=len(a);count+=n;dups+=n-len(np.unique(a,axis=0));g=h[f"{r['robot']}/episode{r['episode']}"];assert np.array_equal(g['state14'][:],a.astype('float32'));assert len(g['frame_index'])==n
   for side in ['left','right']:assert np.array_equal(g[side+'_endpose7'][:],raw['endpose/'+side+'_endpose'][:])
   for cam in ['head_camera','left_camera','right_camera']:assert raw[f'observation/{cam}/rgb'].shape==(n,)
  images+=int(r['images'])
assert count==23814 and images==95256 and dups==5002 and len(seen)==100
arms,bases=setup();accepted=failed=0;perdir=collections.Counter();failtypes=collections.Counter();maxdiff=0
for p in sorted((R/'data/mapping_full').glob('*.json')):
 item=json.loads(p.read_text());a=np.load(R/'data/episodes'/f"{item['robot']}-episode{item['episode']}.npz")['state14'];assert len(item['ik_records'])==2*len(a)
 for r in item['ik_records']:
  perdir[r['source_robot']]+=1
  if r['status']!='kinematic_solution':failed+=1;failtypes[r['status']]+=1;continue
  accepted+=1;side=['left','right'].index(r['side']);s=r['source_robot'];target=r['target_robot'];qs=a[r['frame']].reshape(2,7)[side,:6].astype(float);qt=np.array(r['target_q']);assert np.all(qt>=arms[target][side].chain.limits[:,0]) and np.all(qt<=arms[target][side].chain.limits[:,1])
  goal=bases[s][side]@arms[s][side].chain.fk(qs)@arms[s][side].offset();actual=bases[target][side]@arms[target][side].chain.fk(qt)@arms[target][side].offset();assert np.max(abs(goal-np.array(r['target_tools_world'])))<1e-12
  pos=np.linalg.norm(goal[:3,3]-actual[:3,3])*1000;ang=np.degrees(np.arccos(np.clip((np.trace(goal[:3,:3]@actual[:3,:3].T)-1)/2,-1,1)));assert pos<=.5+1e-9 and ang<=.05+1e-7;maxdiff=max(maxdiff,abs(pos-r['position_mm']))
assert accepted+failed==47628 and failed==87 and failtypes=={'ik_failed':87}
dashboard=json.loads((R/'data/dashboard_data.json').read_text());assert len(dashboard['trajectories'])==100 and dashboard['mapping']['ik_arm_targets']==47628
assert dashboard['summary']['total_frames']==count and sum(t['n'] for t in dashboard['trajectories'])==count
result=dict(passed=True,raw_crc_and_sha_files=100,unique_file_sha=100,canonical_frames=count,decoded_images_from_run=images,recomputed_state_duplicates=dups,mapping_attempts=accepted+failed,mapping_accepted=accepted,mapping_failed=failed,failure_types=dict(failtypes),independent_fk_residual_checks=accepted,max_position_difference_mm=maxdiff,full_execution_mapping_validated=False,elapsed_seconds=round(time.time()-start,2))
(R/'logs/independent-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
