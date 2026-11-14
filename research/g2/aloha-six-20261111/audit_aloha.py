"""Recompute six native results from immutable receipts, arrays and PhysX logs."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import numpy as np


def digest(p):
 h=sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def scan(raw,out,repo):
 out.mkdir(parents=True,exist_ok=True)
 submission=json.loads((raw/'tools/g2-standalone-six-4h-20261111/stage/submission.json').read_text())
 rawhash=json.loads((raw/'RAW-SHA256.json').read_text())
 rows=[];firsts={};slices={};initials={}
 for item in submission['rows']:
  ident=item['experiment'];base=raw/item['result_dir'];trial,=(base/'trials').iterdir()
  receipt=json.loads((trial/'receipt.json').read_text()); events=[json.loads(l) for l in (trial/'events.jsonl').open()]
  config=repo/'research/g2/standalone-six-4h/configs'/f'{ident}.json'
  assert digest(config)==item['config_sha256']
  cfg=json.loads(config.read_text());dest=out/ident;dest.mkdir(exist_ok=True)
  for leaf in ('receipt.json','events.jsonl'):shutil.copyfile(trial/leaf,dest/leaf)
  responses=[e for e in events if e['kind']=='response']
  chunks=sorted(trial.glob('chunk-*.npy'))
  assert len(chunks)==receipt['queries']==len(responses)
  for p,event in zip(chunks,responses):
   assert digest(p)==event['chunk_sha256'];a=np.load(p,allow_pickle=False)
   assert a.shape==(50,14) and np.isfinite(a).all()
  firsts[ident]=np.load(chunks[0],allow_pickle=False)
  counts=Counter();runs=Counter();longest=Counter();first={};last={};first_right_only=None;record=[];n=0;lift=0.
  first_drop=None;post_drop_max=None
  first_contact_impulse={}
  with (trial/'physics.jsonl').open('rb') as f:
   for line_number,line in enumerate(f,1):
    x=json.loads(line);n+=1;tick=x['tick'];assert tick==line_number-1
    if tick==0:initial=x;z0=x['block_pose'][2][3]
    z=x['block_pose'][2][3];lift=max(lift,z-z0)
    touching={name for c in x['contacts'] if c['points'] for name in c['pair']}
    flags=dict(left=bool(touching&{'fl_link7','fl_link8'}),right=bool(touching&{'fr_link7','fr_link8'}),
               left_two={'fl_link7','fl_link8'}<=touching,right_two={'fr_link7','fr_link8'}<=touching,
               target='target_box' in touching)
    flags['both_two']=flags['left_two'] and flags['right_two']
    compact=dict(tick=tick,action=x['action'],block_xyz=[x['block_pose'][i][3] for i in range(3)],
                 dz_m=z-z0,block_linear_velocity=x['block_linear_velocity'],contact_flags=flags,
                 normalized_gripper_command=x.get('normalized_gripper_command'),aperture_m=x.get('aperture_m'),
                 raw_line_number=line_number,raw_line_sha256=sha256(line).hexdigest())
    special=False
    for k,v in flags.items():
     if v:
      if k not in first:first[k]=compact.copy();special=True
      last[k]=compact.copy();counts[k]+=1;runs[k]+=1;longest[k]=max(longest[k],runs[k])
     else:runs[k]=0
    if 'both_two' in first and flags['right_two'] and not flags['left_two'] and first_right_only is None:
     first_right_only=compact.copy();special=True
    if first_right_only is not None and z-z0<-.04 and first_drop is None:
     first_drop=compact.copy();special=True
    if first_drop is not None:post_drop_max=max(z-z0,post_drop_max if post_drop_max is not None else z-z0)
    for side,names in [('left',{'fl_link7','fl_link8'}),('right',{'fr_link7','fr_link8'})]:
     if side not in first_contact_impulse and any(set(c['pair'])&names and any(np.linalg.norm(p['impulse'])>1e-8 for p in c['points']) for c in x['contacts']):
      first_contact_impulse[side]=compact.copy();special=True
    if ident in ('G3','G6') and (special or 11000<=tick<=14500 and tick%100==0 or tick==receipt['physics_steps']):
     compact['contacts']=x['contacts'];compact['block_pose']=x['block_pose'];compact['finger_relative_to_block']=x['finger_relative_to_block'];record.append(compact)
    final=x
  assert n==receipt['physics_steps']+1
  assert len([e for e in events if e['kind']=='action_complete'])==receipt['completed_actions']
  assert receipt['success']==(receipt['outcome']=='SUCCESS')
  assert receipt['scene_closed']
  host=json.loads((base/'standalone-result.json').read_text());assert host['row']['process_group_reaped']
  if not receipt['success']:assert receipt['completed_actions']==cfg['action_cap'] and receipt['outcome']=='TASK_INCOMPLETE'
  write(dest/'initial-physics.json',initial)
  phases=dict(first_contact=first,first_impulse_above_1e_8=first_contact_impulse,first_right_two_without_left_two_after_both=first_right_only,
              contact_tick_counts=dict(counts),longest_contact_run_ticks=dict(longest),last_contact=last,max_lift_m=lift,
              first_40mm_below_start_after_handover=first_drop,post_drop_max_height_relative_to_start_m=post_drop_max,
              final_block_pose=final['block_pose'],final_gripper_command=final.get('normalized_gripper_command'),
              stable_hold_tracker='not recorded in author_native passive observer; two-finger contact is not a load-bearing hold verdict')
  write(dest/'stages.json',phases)
  if record:
   with (dest/'handover-samples.jsonl').open('w') as f:
    for x in record:f.write(json.dumps(x,allow_nan=False)+'\n')
  initials[ident]=initial
  sources={str(p.relative_to(raw)):digest(p) for p in [trial/'receipt.json',trial/'events.jsonl',trial/'physics.jsonl',base/'standalone-result.json',base/'effective-launch.json',base/'authorization-receipt.json',*chunks,*sorted(trial.glob('request-*.npz'))]}
  assert all(rawhash[k]==v for k,v in sources.items())
  rows.append(dict(experiment=ident,job_id=item['job_id'],result_dir=item['result_dir'],scene_seed=receipt['scene_seed'],
                   success=receipt['success'],completed_actions=receipt['completed_actions'],queries=receipt['queries'],
                   physics_steps=receipt['physics_steps'],elapsed_s=receipt['elapsed_s'],outcome=receipt['outcome'],
                   stop_reason='native_success_predicate' if receipt['success'] else 'action_cap_800',
                   config_sha256=digest(config),archive_sha256=item['archive_sha256'],source_sha256=sources,
                   prompt=receipt['prompt'],selector=receipt['selector'],phases=phases))
 pairs=[]
 for left,right in [('G1','G4'),('G2','G5'),('G3','G6')]:
  le=[json.loads(l) for l in (out/left/'events.jsonl').open()];re=[json.loads(l) for l in (out/right/'events.jsonl').open()]
  req=lambda e:next(x['request_sha256'] for x in e if x['kind']=='request_identity')
  rng=lambda e:next(x['rng'] for x in e if x['kind']=='response')
  keys=['qpos','qvel','block_pose','block_linear_velocity','block_angular_velocity','robot_root','target_pose','block_mass_kg']
  pairs.append(dict(pair=[left,right],initial_exact_by_field={k:initials[left][k]==initials[right][k] for k in keys},
                    request_hash_equal=req(le)==req(re),sample_key_equal=rng(le)['sample']==rng(re)['sample'],
                    first_chunk_exact_equal=bool(np.array_equal(firsts[left],firsts[right])),
                    max_abs_first_chunk_delta=np.max(abs(firsts[left]-firsts[right]),axis=0).tolist()))
 write(out/'summary.json',dict(batch_sha256=submission['batch_sha256'],gpu_jobs_submitted_this_round=0,rows=rows,pairs=pairs))
 write(out/'first-chunks.json',{k:v.tolist() for k,v in firsts.items()})
 write(out/'raw-manifest.json',rawhash)
 return rows


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('output',type=Path);p.add_argument('repo',type=Path)
 a=p.parse_args();rows=scan(a.raw,a.output,a.repo)
 for r in rows:print(r['experiment'],r['success'],r['completed_actions'],r['queries'],r['stop_reason'])
