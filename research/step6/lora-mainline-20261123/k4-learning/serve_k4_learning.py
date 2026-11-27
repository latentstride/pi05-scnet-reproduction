"""Serve a matched continuation checkpoint through the original native ARX I/O."""
import argparse
import json
from pathlib import Path
from openpi.serving import websocket_policy_server
from scripts.step5_arx import model, train
from scripts.step5_arx.data import Dataset, digest
from scripts.step5_arx.serve import Session
from scripts.step6_arx.serve import identity_digest, checkpoint_digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('experiment', type=Path)
    p.add_argument('branch', choices=('control','k4_tcp_grip'))
    p.add_argument('--port', type=int, required=True)
    args=p.parse_args()
    identity=json.loads((args.experiment/'identity.json').read_text())
    parent=identity['parent_identity']
    step=identity['updates_per_branch']
    checkpoint=args.experiment/args.branch/'checkpoints'/f'step-{step:04d}'
    candidate_identity=dict(experiment=identity, branch=args.branch)
    train.gpu_device()
    parts, baseline, _=model.load_models(model.validate_install(args.root, full_hash=False))
    graph, initial, frozen, _, opt=parts
    adapted, _=train.restore(checkpoint, initial, opt)
    del parts, baseline
    with Dataset(args.root/'data/step5-arx/arx-drive-target-v1.h5') as data:
        if data.norm_stats != json.loads((args.experiment/'arx-norm.json').read_text()):
            raise ValueError('native normalization differs from this training run')
        policy=model.NativePolicy(data.norm_stats)
        norm_sha=identity_digest(data.norm_stats)
    metadata=dict(variant='lora', dataset_sha256=parent['dataset_sha256'],
        action_space='arx_x5_absolute_drive_target', action_horizon=50, action_dim=14,
        selected_step=step, seed=42, step6_identity=dict(kind='step5_checkpoint',
        pace_revision=parent['pace_checkpoint'], pace_manifest_sha256=parent['manifest_sha256'],
        norm_stats_sha256=norm_sha, training_identity_sha256=identity_digest(candidate_identity),
        selected_checkpoint_sha256=checkpoint_digest(checkpoint)))
    print(json.dumps(metadata), flush=True)
    session=Session(policy, model.sampler(graph, adapted=True), (adapted, frozen), 42)
    websocket_policy_server.WebsocketPolicyServer(session, host='127.0.0.1', port=args.port,
                                                metadata=metadata).serve_forever()


if __name__=='__main__':
    main()
