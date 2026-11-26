"""CPU-only verification of archived metrics, receipts and representative files."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'research/final_20261126'
def read(path):return json.loads(path.read_text())
def check():
 result=read(E/'comparison.json');raw=read(E/'raw-ten-seed-receipts.json')
 nodes=[n for n in result['nodes'] if n['step']==2000]
 rows=sorted([r for n in nodes for r in n['episodes']],key=lambda r:r['seed'])
 assert [r['seed'] for r in rows]==list(range(10))
 assert sum(r['grasped'] for r in rows)==10
 assert sum(r['transferred_to_right'] for r in rows)==8
 assert sum(r['official_check_success'] for r in rows)==7
 assert [r['seed'] for r in rows if r['official_check_success']]==[0,1,2,5,7,8,9]
 for row in rows:
  receipt=raw[str(row['seed'])]
  assert receipt['status']=='finished' and receipt['native_success']==row['official_check_success']
  assert receipt['native_eval_success']==row['official_check_success']
  assert receipt['stop_reason']==('official_success' if row['official_check_success'] else 'native_step_limit')
  assert receipt['actions']<=600
  if row['official_check_success']:
   assert all(abs(x)<30 for x in row['final_target_delta_mm'][:2])
   assert abs(row['final_target_delta_mm'][2])<10 and row['placed_observed']
 with (E/'ten-seed-results.csv').open() as f:csv_rows=list(csv.DictReader(f))
 assert len(csv_rows)==10
 for a,b in zip(csv_rows,rows):
  assert int(a['seed'])==b['seed'] and abs(float(a['stable_lift_mm'])-b['stable_bilateral_max_lift_mm'])<1e-9
 stats=read(E/'dataset150.json');collect=read(E/'collection-manifest.json')
 assert len(collect['accepted'])==150 and all(r['official_check_success'] for r in collect['accepted'])
 seeds={r['seed'] for r in collect['accepted']}
 assert len(seeds)==150 and not seeds.intersection(collect['blocked_original_seeds'])
 assert set(stats['train_seeds'])==seeds and len(stats['splits']['train'])==150
 assert stats['splits']['val']==[7,18,23,31,43] and stats['splits']['test']==[6,29,33,38,48]
 assert read(E/'norm_stats.json')==stats['norm_stats']
 assert read(E/'model-metadata.json')['weights']=='EMA'
 assert read(E/'training-final.json')['adam_count']==2000
 assert read(E/'resource-closeout.json')['gpu']['zero_mib']
 assert read(E/'resource-closeout.json')['platform_verified_status']=='Terminated'
 for entry in read(ROOT/'media/final-arx/manifest.json'):
  p=ROOT/'media/final-arx'/entry['file'];b=p.read_bytes()
  assert len(b)==entry['bytes'] and b[4:8]==b'ftyp'
  assert hashlib.sha256(b).hexdigest()==entry['sha256']
  assert entry['frames']==raw[str(entry['seed'])]['actions']+1
 for source in read(E/'source-provenance.json'):
  p=ROOT/source['archived'];assert hashlib.sha256(p.read_bytes()).hexdigest()==source['archive_sha256']
 print('PASS: 10/10 grasp, 8/10 transfer, 7/10 official; receipts/CSV/splits/stats/media/source hashes agree; Notebook terminated.')
if __name__=='__main__':check()
