"""Differentiable fixed-length form of pinned pi0.sample_actions (10 identical Euler steps)."""
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

def sample_actions(pi,rng,observation,num_steps=10):
 observation=_model.preprocess_observation(None,observation,train=False)
 dt=-1.0/num_steps
 batch_size=observation.state.shape[0]
 noise=jax.random.normal(rng,(batch_size,pi.action_horizon,pi.action_dim))
 prefix_tokens,prefix_mask,prefix_ar=pi.embed_prefix(observation)
 prefix_attn=make_attn_mask(prefix_mask,prefix_ar)
 positions=jnp.cumsum(prefix_mask,axis=1)-1
 _,cache=pi.PaliGemma.llm([prefix_tokens,None],mask=prefix_attn,positions=positions)
 def step(carry,_):
  x,time=carry
  suffix,mask,ar,cond=pi.embed_suffix(observation,x,jnp.broadcast_to(time,batch_size))
  suffix_attn=make_attn_mask(mask,ar)
  prefix_attn=jnp.broadcast_to(prefix_mask[:,None,:],(batch_size,suffix.shape[1],prefix_mask.shape[1]))
  full_mask=jnp.concatenate([prefix_attn,suffix_attn],axis=-1)
  positions=jnp.sum(prefix_mask,axis=-1)[:,None]+jnp.cumsum(mask,axis=-1)-1
  (_,output),_=pi.PaliGemma.llm([None,suffix],mask=full_mask,positions=positions,kv_cache=cache,adarms_cond=[None,cond])
  velocity=pi.action_out_proj(output[:,-pi.action_horizon:])
  return (x+dt*velocity,time+dt),None
 (clean,_),_=jax.lax.scan(jax.checkpoint(step),(noise,1.0),None,length=num_steps)
 return clean

