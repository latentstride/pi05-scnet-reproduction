"""Restricted 3 x 5 LIBERO evaluation using the pinned official control protocol."""
import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time
import traceback

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools, websocket_client_policy


def state(obs):
    q = np.array(obs['robot0_eef_quat'], copy=True)
    q[3] = np.clip(q[3], -1., 1.)
    den = math.sqrt(1. - q[3] ** 2)
    aa = np.zeros(3) if math.isclose(den, 0.) else q[:3] * 2. * math.acos(q[3]) / den
    return np.concatenate((obs['robot0_eef_pos'], aa, obs['robot0_gripper_qpos']))


def image(obs, key='agentview_image'):
    raw = np.ascontiguousarray(obs[key][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(raw, 224, 224))


def atomic_json(path, value):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / 'episodes').mkdir(exist_ok=True)
    (out / 'videos').mkdir(exist_ok=True)
    cfg = {'suite': 'libero_spatial', 'task_order_index': 0, 'task_ids': [0, 1, 2],
           'initial_state_ids': [0, 1, 2, 3, 4], 'seed': 7, 'max_control_steps': 220,
           'settle_steps': 10, 'replan_steps': 5, 'image_size': 224, 'render_resolution': 256,
           'image_rotation_degrees': 180, 'policy': 'pi05_libero', 'openpi_commit': pathlib.Path('source/OPENPI_COMMIT').read_text().strip()}
    atomic_json(out / 'evaluation_config.json', cfg)
    np.random.seed(7)
    suite = benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    rows = []
    fields = ['task_id', 'task', 'initial_state_id', 'seed', 'success', 'termination', 'steps',
              'inference_calls', 'elapsed_seconds', 'video', 'trajectory', 'error']
    for task_id in cfg['task_ids']:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env = OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
        env.seed(7)
        try:
            for initial_id in cfg['initial_state_ids']:
                episode = 'task%d_init%d' % (task_id, initial_id)
                video_path = out / 'videos' / (episode + '.mp4')
                traj_path = out / 'episodes' / (episode + '.npz')
                row = dict(task_id=task_id, task=str(task.language), initial_state_id=initial_id,
                           seed=7, success=False, termination='error', steps=0, inference_calls=0,
                           elapsed_seconds=0., video=str(video_path), trajectory=str(traj_path), error='')
                t0 = time.perf_counter()
                states, executed, chunks, chunk_steps, infer_seconds, rewards, dones = [], [], [], [], [], [], []
                writer = None
                episode_error = None
                try:
                    env.reset()
                    obs = env.set_init_state(initial_states[initial_id])
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.] * 6 + [-1.])
                    actions = collections.deque()
                    states.append(state(obs))
                    writer = imageio.get_writer(str(video_path), fps=10, codec='libx264', quality=7, macro_block_size=16)
                    writer.append_data(image(obs))
                    for step in range(220):
                        if not actions:
                            observation = {'observation/image': image(obs), 'observation/wrist_image': image(obs, 'robot0_eye_in_hand_image'),
                                           'observation/state': state(obs), 'prompt': str(task.language)}
                            assert observation['observation/state'].shape == (8,)
                            begin = time.perf_counter()
                            chunk = np.asarray(client.infer(observation)['actions'])
                            infer_seconds.append(time.perf_counter() - begin)
                            assert chunk.ndim == 2 and chunk.shape[1] == 7 and len(chunk) >= 5 and np.isfinite(chunk).all(), chunk
                            chunks.append(chunk)
                            chunk_steps.append(step)
                            actions.extend(chunk[:5])
                        action = actions.popleft()
                        obs, reward, done, _ = env.step(action.tolist())
                        executed.append(action.copy())
                        states.append(state(obs))
                        rewards.append(reward)
                        dones.append(done)
                        writer.append_data(image(obs))
                        row['steps'] = step + 1
                        if done:
                            row.update(success=True, termination='success')
                            break
                    else:
                        row['termination'] = 'step_limit'
                except Exception:
                    episode_error = traceback.format_exc()
                    row['error'] = episode_error
                finally:
                    if writer is not None:
                        writer.close()
                    row.update(inference_calls=len(chunks), elapsed_seconds=time.perf_counter() - t0)
                    np.savez_compressed(str(traj_path), states=np.asarray(states), executed_actions=np.asarray(executed),
                                        predicted_chunks=np.asarray(chunks), chunk_start_steps=np.asarray(chunk_steps),
                                        inference_seconds=np.asarray(infer_seconds), rewards=np.asarray(rewards), dones=np.asarray(dones))
                    atomic_json(out / 'episodes' / (episode + '.json'), dict(row, initial_state_sha256=hashlib.sha256(np.asarray(initial_states[initial_id]).tobytes()).hexdigest()))
                    rows.append(row)
                    temporary = out / 'evaluation_results.partial.csv'
                    with temporary.open('w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        w.writeheader()
                        w.writerows(rows)
                    temporary.replace(out / 'evaluation_results.csv')
                    atomic_json(out / 'progress.json', {'completed': len(rows), 'expected': 15,
                                'successful': sum(int(r['success']) for r in rows), 'latest': row})
                    print('EPISODE_RESULT', json.dumps(row), flush=True)
                if episode_error:
                    raise RuntimeError(episode_error)
        finally:
            env.close()
    summary = {'episodes': len(rows), 'successful': sum(int(r['success']) for r in rows),
               'success_rate': sum(int(r['success']) for r in rows) / len(rows),
               'per_task': {str(i): {'episodes': 5, 'successful': sum(int(r['success']) for r in rows if r['task_id'] == i)} for i in [0, 1, 2]}}
    assert len(rows) == 15 and all(r['termination'] in ('success', 'step_limit') for r in rows)
    atomic_json(out / 'summary.json', summary)
    subprocess.run([sys.executable, 'scripts/summarize_evaluation.py', str(out)], check=True)
    print('EVALUATION_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
