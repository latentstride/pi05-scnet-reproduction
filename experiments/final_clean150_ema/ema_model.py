"""Fresh PACE base; full image/action experts, only language LoRA; native FM/EMA."""
import dataclasses
import flax.nnx as nnx
from flax.traverse_util import flatten_dict,unflatten_dict
import jax,jax.numpy as jnp,optax,orbax.checkpoint as ocp
from scripts.step5_arx import model as original
from openpi.models import model as openpi_model
NativePolicy=original.NativePolicy
sampler=original.sampler
def trainable(path,value):
 p=tuple(path)
 visual=p[:2]==('PaliGemma','img')
 language_lora=p[:2]==('PaliGemma','llm') and 'lora' in str(p[-1])
 action=p[:2]==('PaliGemma','llm') and any(str(x).endswith('_1') for x in p[2:])
 io=p[0] in ('action_in_proj','action_out_proj','time_mlp_in','time_mlp_out','state_proj','action_time_mlp_in','action_time_mlp_out')
 return issubclass(value.type,nnx.Param) and (visual or language_lora or action or io)
def load(root,with_optimizer=True):
 params_path=original.validate_install(root,full_hash=False)
 _,config=original.model_configs();config=dataclasses.replace(config,action_expert_variant='gemma_300m')
 graph,state=nnx.split(nnx.eval_shape(config.create,jax.random.key(42)))
 spec=flatten_dict(state.to_pure_dict());base=flatten_dict(openpi_model.restore_params(params_path,dtype=jnp.bfloat16))
 keys=jax.random.split(jax.random.key(42),sum('lora' in str(p[-1]) for p in spec));i=0;bound={}
 for p,s in sorted(spec.items()):
  if 'lora' in str(p[-1]):
   bound[p]=jax.random.normal(keys[i],s.shape,dtype=jnp.float32)*.01 if p[-1].endswith('_a') else jnp.zeros(s.shape,jnp.float32);i+=1
  else:
   assert p in base and base[p].shape==s.shape,(p,s.shape)
   bound[p]=base[p]
 state.replace_by_pure_dict(unflatten_dict(bound));model=nnx.merge(graph,state)
 graph,parameters,frozen=nnx.split(model,trainable,...);parameters=jax.tree.map(lambda a:a.astype(jnp.float32),parameters)
 entries=[dict(path='/'.join(p),elements=v.size,shape=list(v.shape)) for p,v in sorted(flatten_dict(parameters.to_pure_dict()).items())]
 assert any(e['path'].startswith('PaliGemma/img/') for e in entries)
 assert not any(p[:2]==('PaliGemma','llm') and not any(str(x).endswith('_1') for x in p[2:]) and 'lora' not in str(p[-1]) for p in flatten_dict(parameters.to_pure_dict()))
 optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(1.,weight_decay=1e-2)) if with_optimizer else None
 opt=optimizer.init(parameters) if optimizer is not None else None
 return graph,parameters,frozen,optimizer,opt,entries
def restore(path,template):
 shard=jax.sharding.SingleDeviceSharding(jax.devices()[0])
 args=jax.tree.map(lambda x:ocp.ArrayRestoreArgs(sharding=shard,restore_type=jax.Array,dtype=x.dtype),template)
 with ocp.PyTreeCheckpointer() as reader:
  return reader.restore(path,args=ocp.args.PyTreeRestore(item=template,transforms={},restore_args=args))
def save(path,online,ema,opt):
 if path.exists():raise FileExistsError(path)
 item=dict(online=online.to_pure_dict(),ema=ema.to_pure_dict(),optimizer=jax.tree.leaves(opt))
 # PyTreeCheckpointer.save is synchronous in the installed Orbax version.
 # wait_until_finished belongs to AsyncCheckpointer, not PyTreeCheckpointer.
 with ocp.PyTreeCheckpointer() as writer:writer.save(path,args=ocp.args.PyTreeSave(item))
