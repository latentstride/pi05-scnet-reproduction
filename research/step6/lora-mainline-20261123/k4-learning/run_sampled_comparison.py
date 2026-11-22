"""Replay the completed treatment's exact training draws with only label canonicalization changed."""
import json
from pathlib import Path
import sys
import time
print('Loading matched label-repair training',flush=True)
import jax
import jax.numpy as jnp
import numpy as np
from scripts.step5_arx import model,train
from scripts.step5_arx.config import write_json
from scripts.step5_arx.data import Dataset,ALOHA_SCHEMA,digest
from g2_assets import load_arms
from k4_objective import make_auxiliary
from run_k4_learning import validate
from run_sampled_k4 import optimizer_step
from sampled_actions import sample_actions
import flax.nnx as nnx
from unique_observations import relabel

root,baseline,mapping_path,out=map(Path,sys.argv[1:5]);out.mkdir()
mapping=json.loads(mapping_path.read_text())
original=json.loads((baseline/'identity.json').read_text())
if mapping['source_sha256']!=original['parent_identity']['dataset_sha256']:
    raise ValueError('mapping and original dataset differ')
draws=[json.loads(line) for line in (baseline/'k4_tcp_grip/train.jsonl').read_text().splitlines()]
identity=dict(parent_comparison=original,parent_comparison_path=str(baseline),
    experiment='matched_actual_sampling_K4_auxiliary',reference_label_repair_run='step5-k4-unique-20261122T192056Z',auxiliary='same coefficient/scales; ten-step sampled actions instead of single-flow clean estimate',mapping_sha256=digest(mapping_path),
    unchanged='initial checkpoint, fresh Adam, input observations, draws, flow RNG, norm, auxiliary coefficient/scales, batch and split',
    target_values='original expert commands only, next unique observation-event order',
    physical_timestamps='unknown',updates=len(draws),
    source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
write_json(out/'identity.json',identity)
bundle=root/'tools/a800-closed-loop'
safety=json.loads((bundle/'suite/docs/step6/arx_safety_profile.json').read_text())
arms=load_arms(bundle/'PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/assets',
              bundle/'pr5/research/g2/g2e-20261106-r1/release/asset-audit.json','ARX-X5',safety['calibration'])
parts,old,_=model.load_models(model.validate_install(root,full_hash=False));del old
graph,initial,frozen,optimizer,initial_opt=parts
fitted,_=train.restore(Path(original['parent_run'])/'checkpoints/step-0600',initial,initial_opt)
opt=optimizer.init(fitted)
pred=model.sampler(graph,adapted=True)
with Dataset(root/'data/step5-arx/arx-drive-target-v1.h5') as arx,Dataset(
        root/'data/step5-embodiment-mix/aloha-drive-target-v1.h5',schema=ALOHA_SCHEMA) as aloha:
    datasets=[arx,aloha];policies=[model.NativePolicy(d.norm_stats) for d in datasets]
    step=optimizer_step(graph,optimizer,make_auxiliary(arms,arx.norm_stats))
    sample=relabel(arx,mapping,'train',*draws[0]['windows'][0])
    check_obs,_=policies[0].batch([sample])
    @jax.jit
    def equivalence(parameters,fixed,obs):
        pi=nnx.merge(graph,parameters,fixed);pi.eval()
        key=jax.random.fold_in(jax.random.key(42),601)
        official=pi.sample_actions(key,obs,num_steps=10)
        copied=sample_actions(pi,key,obs,10)
        return jnp.max(abs(official-copied))
    difference=float(equivalence(fitted,frozen,check_obs))
    write_json(out/'sampler-equivalence.json',dict(max_abs_difference=difference,steps=10))
    print('SAMPLER_EQUIVALENCE',difference,flush=True)
    if not np.isfinite(difference) or difference>1e-5:
        raise ValueError('differentiable sampling differs from pinned inference')
    started=time.monotonic()
    for draw in draws:
        i=draw['step'];index=draw['sampler_step'];body=draw['body'];windows=draw['windows']
        if body==0:samples=[relabel(arx,mapping,'train',ep,t) for ep,t in windows]
        else:samples=[aloha.read('train',ep,t,terminal=True) for ep,t in windows]
        obs,target=policies[body].batch(samples)
        fitted,opt,metrics=step(fitted,frozen,opt,jax.random.fold_in(jax.random.key(42),index),obs,target,
            np.asarray([s['valid_rows'] for s in samples],np.int32),np.stack([s['state'] for s in samples]),
            body==0)
        metrics=np.asarray(metrics)
        if not np.isfinite(metrics).all():raise ValueError(('nonfinite update',i,metrics))
        record=dict(step=i,sampler_step=index,body=body,windows=windows,
            metrics=dict(zip(('total','grad','update','flow','auxiliary','tcp','rotation','grip'),map(float,metrics))),
            elapsed_seconds=time.monotonic()-started,
            target_raw_indices=[s.get('target_raw_indices') for s in samples])
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        write_json(out/'progress.json',record)
        print(json.dumps({k:record[k] for k in ('step','body','metrics','elapsed_seconds')}),flush=True)
    (out/'checkpoints').mkdir()
    checkpoint=out/'checkpoints'/f'step-{len(draws):04d}'
    train.save(checkpoint,fitted,opt)
    # Evaluate against the UNCHANGED raw validation labels for comparability.
    validate(arx,policies[0],pred,(fitted,frozen),arms,out/'raw-val')
    write_json(out/'complete.json',dict(updates=len(draws),status='matched_sampled_auxiliary_fit_and_raw_val_complete',
        physical_execution=False))
    from scripts.step6_arx.serve import identity_digest,checkpoint_digest
    from scripts.step5_arx.serve import Session
    from openpi.serving import websocket_policy_server
    parent=original['parent_identity']
    metadata=dict(variant='lora',dataset_sha256=parent['dataset_sha256'],action_space='arx_x5_absolute_drive_target',
        action_horizon=50,action_dim=14,selected_step=len(draws),seed=42,step6_identity=dict(kind='step5_checkpoint',
        pace_revision=parent['pace_checkpoint'],pace_manifest_sha256=parent['manifest_sha256'],
        norm_stats_sha256=identity_digest(arx.norm_stats),training_identity_sha256=identity_digest(identity),
        selected_checkpoint_sha256=checkpoint_digest(checkpoint)))
    write_json(out/'warm-service.json',dict(port=51925,metadata=metadata))
    session=Session(policies[0],pred,(fitted,frozen),42)
    websocket_policy_server.WebsocketPolicyServer(session,host='127.0.0.1',port=51925,metadata=metadata).serve_forever()

