"""Matched auxiliary objective on actual ten-step inference outputs."""
from functools import partial
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from k4_objective import flow_and_clean
from sampled_actions import sample_actions

def optimizer_step(graph,optimizer,auxiliary):
 @partial(jax.jit,static_argnames=('native',))
 def step(adapters,frozen,opt_state,key,obs,target,valid,raw_state,native):
  def objective(parameters):
   pi=nnx.merge(graph,parameters,frozen);pi.train()
   flow,_=flow_and_clean(pi,key,obs,target)
   row=jnp.arange(50);weight=jnp.where(row<4,1.,4/46)*(row<valid[:,None])
   flow=jnp.mean(jnp.sum(flow*weight,axis=-1)/jnp.sum(weight,axis=-1))
   if native:
    pi.eval()
    sampled=sample_actions(pi,jax.random.fold_in(key,1),obs,10)
    extra,components=auxiliary(sampled,target,raw_state,valid)
   else:extra=jnp.float32(0.);components=jnp.zeros(3)
   return flow+.1*extra,jnp.concatenate((jnp.array([flow,extra]),components))
  (value,parts),grad=jax.value_and_grad(objective,has_aux=True)(adapters)
  updates,opt_state=optimizer.update(grad,opt_state,adapters)
  return optax.apply_updates(adapters,updates),opt_state,jnp.concatenate((jnp.array([value,optax.global_norm(grad),optax.global_norm(updates)]),parts))
 return step

