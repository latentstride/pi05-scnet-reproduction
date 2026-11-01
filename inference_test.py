"""One GPU inference using the pinned official pi05_libero policy.

Without --observation this uses the official synthetic example, not a simulator observation.
An observation NPZ must contain observation/state, observation/image,
observation/wrist_image and prompt; pickle loading is disabled.
"""
import argparse
import importlib.metadata
import json
import pathlib
import subprocess
import time

import jax
import numpy as np
from openpi.policies import libero_policy, policy_config
from openpi.training import config


def gpu_snapshot():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used',
        '--format=csv,noheader',
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
    parser.add_argument('--observation', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('results/inference'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if not any(d.platform == 'gpu' for d in devices):
        raise RuntimeError(f'No JAX GPU: {devices}')
    np.random.seed(0)
    if args.observation:
        with np.load(args.observation, allow_pickle=False) as data:
            observation = {key: data[key] for key in data.files}
        observation['prompt'] = str(observation['prompt'].item())
        source = str(args.observation)
    else:
        observation = libero_policy.make_libero_example()
        source = 'official_synthetic_example'
    assert observation['observation/state'].shape == (8,)
    for key in ('observation/image', 'observation/wrist_image'):
        assert observation[key].shape == (224, 224, 3)
        assert observation[key].dtype == np.uint8
    report = {'observation_source': source, 'seed': 0, 'devices': [str(d) for d in devices],
              'gpu_before': gpu_snapshot(), 'checkpoint': str(args.checkpoint),
              'versions': {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'torch', 'numpy')}}
    trained_config = config.get_config('pi05_libero')
    start = time.perf_counter()
    policy = policy_config.create_trained_policy(trained_config, args.checkpoint)
    report['load_seconds'] = time.perf_counter() - start
    report['gpu_after_load'] = gpu_snapshot()
    print('MODEL_LOADED', report['load_seconds'], flush=True)
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        actions = np.asarray(policy.infer(observation)['actions'])
        timings.append(time.perf_counter() - start)
        assert actions.shape == (trained_config.model.action_horizon, 7), actions.shape
        assert np.isfinite(actions).all()
    report.update(inference_seconds=timings, action_shape=list(actions.shape),
                  action_min=float(actions.min()), action_max=float(actions.max()),
                  finite=True, gpu_after_inference=gpu_snapshot())
    np.savez_compressed(args.output / 'observation_and_actions.npz', **observation, actions=actions)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
