"""Native OpenPI compute_loss; equal H50 MSE; trainable-parameter EMA after optimizer step."""
import json,sys,time,math,gc,shutil
from pathlib import Path
import flax.nnx as nnx
import jax,jax.numpy as jnp,numpy as np,optax
import ema_model as model
from dataset150 import Dataset
from sampling_six import SixPhaseSampler
from augmentation import perturb_samples
from lifecycle import write
root,out=map(Path,sys.argv[1:3]);start,end=map(int,sys.argv[3:5])
assert (start,end) in ((0,2000),(2000,3000))
def lr(count):
 step=jnp.asarray(count)+1.;p=jnp.clip((step-100.)/2900.,0.,1.)
 return jnp.where(step<=100.,3e-5*step/100.,3e-6+.5*(3e-5-3e-6)*(1+jnp.cos(jnp.pi*p)))
write(out/'training-progress.json',dict(stage='loading_fresh_PACE',start=start,end=end))
graph,params,frozen,tx,opt,entries=model.load(root)
ema=jax.tree.map(lambda a:a,params)
if start:
 template=dict(online=params.to_pure_dict(),ema=ema.to_pure_dict(),optimizer=jax.tree.leaves(opt))
 saved=model.restore(out/f'checkpoints/step-{start:04d}',template)
 params.replace_by_pure_dict(saved['online']);ema.replace_by_pure_dict(saved['ema']);opt=jax.tree.unflatten(jax.tree.structure(opt),saved['optimizer']);del saved
assert int(opt[1][0].count)==start
write(out/'trainable-parameters.json',dict(leaves=entries,elements=sum(x['elements'] for x in entries),vision='full PaliGemma/img',language='gemma_2b LoRA only',action='full gemma300m, projections/time MLP'))
write(out/f'restore-{start:04d}.json',dict(start=start,fresh_base=start==0,ema_restored=start>0,optimizer_count=int(opt[1][0].count)))
@jax.jit
def gradient(p,f,key,obs,target):
 def loss(v):
  pi=nnx.merge(graph,v,f);pi.train()
  return jnp.mean(pi.compute_loss(key,obs,target,train=True))
 return jax.value_and_grad(loss)(p)
@jax.jit
def add(a,b):return jax.tree.map(lambda x,y:x+y,a,b)
@jax.jit
def apply(p,e,opt,total):
 g=jax.tree.map(lambda a:a/4,total);rate=lr(opt[1][0].count)
 u,opt=tx.update(g,opt,p);u=jax.tree.map(lambda a:rate*a,u);p=optax.apply_updates(p,u)
 e=jax.tree.map(lambda old,new:.99*old+.01*new,e,p)
 return p,e,opt,jnp.array([optax.global_norm(g),optax.global_norm(u),rate])
with Dataset(root/'data/step5-arx-clean150-newnorm-20261126-v1.h5') as ds:
 policy=model.NativePolicy(ds.norm_stats);sampler=SixPhaseSampler(ds);pool=tuple(ds.indices('train',terminal=True))
 write(out/'sampling.json',sampler.manifest());beg=time.monotonic()
 for step in range(start+1,end+1):
  total=None;losses=[]
  for micro in range(4):
   index=(step-1)*4+micro+1;windows=sampler.batch(42,index,4,pool)
   samples=[ds.read('train',ep,t,terminal=True) for ep,t in windows];samples,_=perturb_samples(samples,index)
   obs,target=policy.batch(samples)
   if step==start+1:write(out/'training-progress.json',dict(stage='compiling_first_update',step=step,micro=micro))
   loss,g=gradient(params,frozen,jax.random.fold_in(jax.random.key(42),index),obs,target)
   total=g if total is None else add(total,g);losses.append(float(loss))
  params,ema,opt,norms=apply(params,ema,opt,total);norms=np.asarray(norms)
  assert int(opt[1][0].count)==step
  if not np.isfinite(losses).all() or not np.isfinite(norms).all():raise FloatingPointError('nonfinite flow update')
  record=dict(stage='training',step=step,adam_count=int(opt[1][0].count),flow_loss=float(np.mean(losses)),grad_norm=float(norms[0]),update_norm=float(norms[1]),learning_rate=float(norms[2]),ema_decay=.99,elapsed_s=time.monotonic()-beg)
  write(out/'training-progress.json',record)
  with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  if step%10==0 or step==start+1:print(json.dumps(record),flush=True)
 # Each checkpoint explicitly contains online+optimizer for continuity and EMA for every evaluation.
 (out/'checkpoints').mkdir(exist_ok=True)
 required=sum(a.size*a.dtype.itemsize for a in jax.tree.leaves((params,ema,opt)))+2*2**30
 if shutil.disk_usage(out).free<required:raise OSError(f'Insufficient space for checkpoint: need {required} bytes; protected files retained')
 model.save(out/f'checkpoints/step-{end:04d}',params,ema,opt)
 write(out/f'ema-{end:04d}.json',dict(step=end,ema_decay=.99,evaluation_weights='ema',online_and_adam_preserved=True))
 print('EMA_CHECKPOINT_SAVED',end,flush=True)
