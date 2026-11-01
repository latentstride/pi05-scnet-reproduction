"""Capture a real LIBERO observation with the pinned OpenPI preprocessing."""
import argparse
import json
import math
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('results/observation'))
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task = suite.get_task(0)
    bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(10):
            obs, reward, done, info = env.step([0.] * 6 + [-1.])
        quat = np.array(obs['robot0_eef_quat'], copy=True)
        quat[3] = np.clip(quat[3], -1., 1.)
        den = math.sqrt(1. - quat[3] ** 2)
        axisangle = np.zeros(3) if math.isclose(den, 0.) else quat[:3] * 2. * math.acos(quat[3]) / den
        state = np.concatenate((obs['robot0_eef_pos'], axisangle, obs['robot0_gripper_qpos']))
        def image(key):
            raw = np.ascontiguousarray(obs[key][::-1, ::-1])
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(raw, 224, 224))
        record = {'observation/state': state, 'observation/image': image('agentview_image'),
                  'observation/wrist_image': image('robot0_eye_in_hand_image'), 'prompt': str(task.language)}
        assert state.shape == (8,) and np.isfinite(state).all()
        np.savez_compressed(args.output / 'libero_observation.npz', **record)
        report = {'suite': 'libero_spatial', 'task_id': 0, 'initial_state_id': 0,
                  'seed': args.seed, 'settle_steps': 10, 'prompt': str(task.language),
                  'bddl': str(bddl), 'state_shape': list(state.shape), 'image_shape': [224, 224, 3],
                  'image_rotation_degrees': 180, 'dummy_actions_sent': 10, 'done_after_settle': bool(done)}
        (args.output / 'capture.json').write_text(json.dumps(report, indent=2))
        print('LIBERO_OBSERVATION_OK', json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
