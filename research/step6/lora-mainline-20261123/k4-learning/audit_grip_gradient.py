"""Measure the actual grip auxiliary gradient and its effect on a LoRA update."""
import json
from pathlib import Path
import sys
print('Loading gradient diagnostic environment',flush=True)
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
from scripts.step5_arx import model,train
from scripts.step5_arx.data import Dataset
from scripts.step5_arx.sampling import GraspSampler
from scripts.step5_arx.config import write_json
from g2_assets import load_arms
from k4_objective import flow_and_clean,make_auxiliary

root=Path(sys.argv[1]);experiment=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir()
bundle=root/'tools/a800-closed-loop'
safety=json.loads((bundle/'suite/docs/step6/arx_safety_profile.json').read_text())
arms=load_arms(bundle/'PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/assets',
              bundle/'pr5/research/g2/g2e-20261106-r1/release/asset-audit.json','ARX-X5',safety['calibration'])
print('Loading parent and candidate adapters',flush=True)
parts,baseline,_=model.load_models(model.validate_install(root,full_hash=False));del baseline
graph,initial,frozen,optimizer,initial_opt=parts
parent=root/'results/step5-arx-aloha-execution-20261122T085555Z/checkpoints/step-0600'
parent_state,_=train.restore(parent,initial,initial_opt)
candidate,_=train.restore(experiment/'k4_tcp_grip/checkpoints/step-0100',initial,initial_opt)
with Dataset(root/'data/step5-arx/arx-drive-target-v1.h5') as dataset:
    policy=model.NativePolicy(dataset.norm_stats);sampler=GraspSampler(dataset)
    auxiliary=make_auxiliary(arms,dataset.norm_stats)
    pools=tuple(dataset.indices('train',terminal=True))
    @jax.jit
    def audit(parameters,fixed,opt,key,obs,target,raw,valid):
        def components(p):
            pi=nnx.merge(graph,p,fixed);pi.train()
            flow,clean=flow_and_clean(pi,key,obs,target)
            row=jnp.arange(50);w=jnp.where(row<4,1.,4/46)*(row<valid[:,None])
            flow=jnp.mean(jnp.sum(flow*w,axis=-1)/jnp.sum(w,axis=-1))
            extra,values=auxiliary(clean,target,raw,valid)
            return jnp.concatenate((jnp.array([flow]),values))
        def total(p):
            c=components(p);return c[0]+.1*jnp.sum(c[1:]),c
        (loss,values),g_total=jax.value_and_grad(total,has_aux=True)(parameters)
        grip_loss,g_grip=jax.value_and_grad(lambda p:components(p)[3])(parameters)
        g_without=jax.tree.map(lambda g,h:g-.1*h,g_total,g_grip)
        change,_=optimizer.update(g_total,opt,parameters)
        without,_=optimizer.update(g_without,opt,parameters)
        difference=jax.tree.map(lambda x,y:x-y,change,without)
        leaf_norms=jnp.stack([jnp.linalg.norm(x) for x in jax.tree.leaves(g_grip)])
        return dict(total=loss,components=values,grip_loss=grip_loss,
                    total_gradient_norm=optax.global_norm(g_total),grip_gradient_norm=optax.global_norm(g_grip),
                    weighted_grip_gradient_norm=.1*optax.global_norm(g_grip),
                    update_norm=optax.global_norm(change),
                    update_difference_without_grip=optax.global_norm(difference),
                    grip_gradient_leaf_norms=leaf_norms)
    reports=[]
    for index in (601,610):
        windows=sampler.batch(42,index,4,pools)
        samples=[dataset.read('train',ep,row,terminal=True) for ep,row in windows]
        obs,target=policy.batch(samples)
        for label,parameters in [('parent',parent_state),('candidate',candidate)]:
            r=audit(parameters,frozen,optimizer.init(parameters),jax.random.fold_in(jax.random.key(42),index),
                    obs,target,np.stack([s['state'] for s in samples]),np.array([s['valid_rows'] for s in samples],np.int32))
            r={k:np.asarray(v).tolist() for k,v in r.items()}
            r.update(model=label,sampler_step=index,windows=windows,
                     normalized_grip_labels=target[:,:4,[6,13]].tolist(),
                     physical_grip_labels=[s['actions'][:4,[6,13]].tolist() for s in samples],
                     valid_rows=[s['valid_rows'] for s in samples],
                     optimizer_context='fresh; parent step601 reproduces actual treatment first update')
            reports.append(r);write_json(out/'gradient.json',reports)
            print(json.dumps({k:r[k] for k in ('model','sampler_step','total','components','total_gradient_norm',
                'grip_gradient_norm','weighted_grip_gradient_norm','update_norm','update_difference_without_grip')}),flush=True)
write_json(out/'complete.json',dict(status='gradient_audit_complete',training_updates_saved=False))
