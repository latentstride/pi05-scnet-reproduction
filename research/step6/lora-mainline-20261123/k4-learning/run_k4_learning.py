"""Matched, bounded continuation of existing mixed LoRA; original labels/split.

Uses the existing execution-aligned training modules without editing them.
Only treatment adds an ARX first-K4 decoded TCP/gripper loss. Both branches
start from the same step-0600 and fresh Adam state, with identical draws/RNG.
"""
import argparse
import faulthandler
faulthandler.enable()
print('Loading existing training environment', flush=True)
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from scripts.step5_arx import model, train
from scripts.step5_arx.config import SETTINGS, write_json
from scripts.step5_arx.data import Dataset, ALOHA_SCHEMA, digest
from scripts.step5_arx.sampling import GraspSampler, grasp_events, sample_aloha
from g2_assets import load_arms
from g2_kinematics import pose_error
from k4_objective import flow_and_clean, make_auxiliary
from k4_fk import verify_fk


def optimizer_step(graph, optimizer, auxiliary):
    @jax.jit
    def step(adapters, frozen, opt_state, key, obs, target, valid, raw_state, coefficient):
        def objective(parameters):
            pi = nnx.merge(graph, parameters, frozen)
            pi.train()
            flow, clean = flow_and_clean(pi, key, obs, target)
            row = jnp.arange(50)
            weight = jnp.where(row < 4, 1., 4/46)*(row < valid[:, None])
            flow = jnp.mean(jnp.sum(flow*weight, axis=-1)/jnp.sum(weight, axis=-1))
            extra, components = auxiliary(clean, target, raw_state, valid)
            return flow+coefficient*extra, jnp.concatenate((jnp.array([flow, extra]), components))
        (value, parts), gradients = jax.value_and_grad(objective, has_aux=True)(adapters)
        updates, opt_state = optimizer.update(gradients, opt_state, adapters)
        return optax.apply_updates(adapters, updates), opt_state, jnp.concatenate(
            (jnp.array([value, optax.global_norm(gradients), optax.global_norm(updates)]), parts))
    return step



def validate(dataset, policy, predict, state, arms, output):
    records = []
    output.mkdir()
    for ep, group in sorted(dataset.groups['val'].items()):
        close, lift = grasp_events(group['drive_target'][:])
        for phase, row in [('approach', close-8), ('pregrasp', close-2), ('close', close), ('lift', lift-2)]:
            sample = dataset.read('val', ep, row)
            obs, target = policy.batch([sample])
            key = jax.random.fold_in(jax.random.key(SETTINGS.seed+1), ep*10000+row)
            _, key = jax.random.split(key)
            raw = np.asarray(predict(state, key, obs))[0]
            pred = policy.adapter.outputs(sample['state'], raw)
            expert = sample['actions']
            def first(g):
                indices = np.flatnonzero(g <= .9)
                return int(indices[0]) if len(indices) else None
            errors = []
            for p, e in zip(pred[:4], expert[:4]):
                errors.append([pose_error(arm.chain.fk(e[s*7:s*7+6])@arm.offset(),
                                          arm.chain.fk(p[s*7:s*7+6])@arm.offset())
                               for s, arm in enumerate(arms)])
            errors = np.asarray(errors)
            record = dict(episode=ep, row=row, phase=phase,
                tcp_mm=(np.linalg.norm(errors[..., :3], axis=-1)*1000).tolist(),
                tcp_deg=np.degrees(np.linalg.norm(errors[..., 3:], axis=-1)).tolist(),
                left_grip_k4=pred[:4, 6].tolist(), expert_grip_k4=expert[:4, 6].tolist(),
                first_close_model=first(pred[:, 6]), first_close_expert=first(expert[:, 6]),
                k4_grip_mae=float(np.mean(abs(pred[:4, [6, 13]]-expert[:4, [6, 13]]))))
            records.append(record)
            np.savez(output/f'ep{ep:02d}-{phase}.npz', raw_prediction=raw,
                     decoded_prediction=pred, expert=expert, state=sample['state'])
            write_json(output/'metrics.json', records)
    phases = {p: [r for r in records if r['phase']==p] for p in ('approach','pregrasp','close','lift')}
    summary = {p: dict(left_tcp_mean_mm=float(np.mean([np.array(r['tcp_mm'])[:, 0] for r in rows])),
                       left_tcp_max_mm=float(np.max([np.array(r['tcp_mm'])[:, 0] for r in rows])),
                       grip_mae=float(np.mean([r['k4_grip_mae'] for r in rows])),
                       closure_rows=[r['first_close_model'] for r in rows]) for p, rows in phases.items()}
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(stage='validation', path=str(output), metrics=summary)), flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--serve-port', type=int, default=0)
    args=parser.parse_args()
    root=args.root; out=args.output
    out.mkdir()
    parent=root/'results/step5-arx-aloha-execution-20261122T085555Z'
    bundle=root/'tools/a800-closed-loop'
    safety=json.loads((bundle/'suite/docs/step6/arx_safety_profile.json').read_text())
    arms=load_arms(bundle/'PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/assets',
                   bundle/'pr5/research/g2/g2e-20261106-r1/release/asset-audit.json', 'ARX-X5', safety['calibration'])
    print('Checking differentiable FK precision', flush=True)
    write_json(out/'fk-regression.json', verify_fk(arms))
    print('FK regression passed', flush=True)
    identity=dict(parent_run=str(parent), parent_step=600, parent_identity=json.loads((parent/'identity.json').read_text()),
        parent_metadata_sha256=digest(parent/'checkpoints/step-0600/_CHECKPOINT_METADATA'),
        experiment='matched_flow_vs_ARX_K4_TCP_gripper_supervision', updates_per_branch=args.steps,
        batch_size=4, seed=42, optimizer='fresh AdamW, same lr/clip as parent',
        sampler='unchanged execution-aligned sampler, step indices 601 onward',
        treatment=dict(coefficient=.1, position_scale_m=.05, rotation_scale_rad=.35,
                       rows=4, bodies='ARX auxiliary; Aloha original flow only'),
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        selection_split='val', test_used_for_training=False, expert_labels_modified=False)
    write_json(out/'identity.json', identity)
    train.gpu_device()
    print('Loading existing step-0600 weights', flush=True)
    parts, baseline, _=model.load_models(model.validate_install(root, full_hash=False))
    del baseline
    graph, initial, frozen, optimizer, initial_opt=parts
    parent_train, _=train.restore(parent/'checkpoints/step-0600', initial, initial_opt)
    pred=model.sampler(graph, adapted=True)
    with Dataset(root/'data/step5-arx/arx-drive-target-v1.h5') as arx, Dataset(
            root/'data/step5-embodiment-mix/aloha-drive-target-v1.h5', schema=ALOHA_SCHEMA) as aloha:
        datasets=[arx, aloha]; policies=[model.NativePolicy(d.norm_stats) for d in datasets]
        samplers=[GraspSampler(d) for d in datasets]
        pools=[tuple(d.indices('train', terminal=True)) for d in datasets]
        for dataset, name in zip(datasets, ['arx','aloha']):
            write_json(out/f'{name}-norm.json', dataset.norm_stats)
        step=optimizer_step(graph, optimizer, make_auxiliary(arms, arx.norm_stats))
        # Confirm the copied forward produces exactly the pinned loss on a real batch.
        sample=arx.read('train', *samplers[0].window(42, 602), terminal=True)
        obs, target=policies[0].batch([sample])
        @jax.jit
        def equivalence(adapters, fixed, obs, target):
            pi=nnx.merge(graph, adapters, fixed);pi.eval()
            key=jax.random.key(42)
            native=pi.compute_loss(key, obs, target, train=False)
            copied, _=flow_and_clean(pi, key, obs, target, train=False)
            return jnp.max(abs(native-copied))
        difference=float(equivalence(parent_train, frozen, obs, target))
        write_json(out/'flow-equivalence.json', dict(max_abs_difference=difference))
        if not np.isfinite(difference) or difference>1e-5:
            raise ValueError('copied forward differs from pinned original')
        validate(arx, policies[0], pred, (parent_train, frozen), arms, out/'parent-val')
        for branch, coefficient in [('control', 0.), ('k4_tcp_grip', .1)]:
            fitted=jax.tree.map(lambda v:v, parent_train)
            opt=optimizer.init(fitted)
            directory=out/branch; directory.mkdir();(directory/'checkpoints').mkdir()
            start=time.monotonic()
            for i in range(1, args.steps+1):
                index=600+i; body=int(sample_aloha(index))
                windows=samplers[body].batch(42, index, 4, pools[body])
                samples=[datasets[body].read('train', ep, row, terminal=True) for ep, row in windows]
                obs, target=policies[body].batch(samples)
                fitted,opt,metrics=step(fitted, frozen, opt, jax.random.fold_in(jax.random.key(42), index),
                    obs, target, np.asarray([s['valid_rows'] for s in samples], np.int32),
                    np.stack([s['state'] for s in samples]), jnp.float32(coefficient if body==0 else 0.))
                metrics=np.asarray(metrics)
                if not np.isfinite(metrics).all():raise ValueError(('nonfinite update', branch, i, metrics))
                record=dict(branch=branch, step=i, sampler_step=index, body=body, windows=windows,
                            metrics=dict(zip(('total','grad','update','flow','auxiliary','tcp','rotation','grip'), map(float, metrics))),
                            elapsed_seconds=time.monotonic()-start)
                with (directory/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                write_json(out/'progress.json', record)
                print(json.dumps(record), flush=True)
            checkpoint=directory/'checkpoints'/f'step-{args.steps:04d}'
            train.save(checkpoint, fitted, opt)
            validate(arx, policies[0], pred, (fitted, frozen), arms, directory/'val')
            if branch == 'control':
                del fitted,opt
        write_json(out/'complete.json', dict(status='matched_training_and_val_complete',
            physical_execution=False, steps_per_branch=args.steps, test_evaluated=False))
        if args.serve_port:
            from openpi.serving import websocket_policy_server
            from scripts.step5_arx.serve import Session
            from scripts.step6_arx.serve import identity_digest, checkpoint_digest
            parent_identity=identity['parent_identity']
            candidate=dict(experiment=identity, branch='k4_tcp_grip')
            metadata=dict(variant='lora', dataset_sha256=parent_identity['dataset_sha256'],
                action_space='arx_x5_absolute_drive_target', action_horizon=50, action_dim=14,
                selected_step=args.steps, seed=42, step6_identity=dict(kind='step5_checkpoint',
                pace_revision=parent_identity['pace_checkpoint'], pace_manifest_sha256=parent_identity['manifest_sha256'],
                norm_stats_sha256=identity_digest(arx.norm_stats), training_identity_sha256=identity_digest(candidate),
                selected_checkpoint_sha256=checkpoint_digest(checkpoint)))
            write_json(out/'warm-service.json', dict(port=args.serve_port, metadata=metadata))
            session=Session(policies[0], pred, (fitted, frozen), 42)
            websocket_policy_server.WebsocketPolicyServer(session, host='127.0.0.1', port=args.serve_port,
                                                         metadata=metadata).serve_forever()


if __name__=='__main__':
    main()
