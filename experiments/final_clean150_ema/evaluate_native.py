"""Independent RoboTwin qpos evaluation. No G2 controller or intervention.

Only two runtime setup accommodations: initialize the unused Cartesian planner
with RoboTwin's own MplibPlanner (CuRobo is absent), and compensate gravity on
every scene step, including initialization. take_action/check_success unmodified.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, default=serial, indent=2) + '\n')
    tmp.replace(path)


def main():
    chunk=int(os.environ["ACTION_CHUNK_K"])
    if chunk not in (4,8,15,25):raise ValueError("K must be4/8/15/25")
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('seed', type=int)
    parser.add_argument('--port', type=int, default=52060)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root, out = args.root, args.out
    pace = root / 'tools/a800-closed-loop/PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1'
    os.chdir(pace)
    sys.path[:0] = [str(pace), str(pace / 'pace/src'), str(pace / 'description/utils')]
    # robot.py unconditionally imports this optional class. It is never invoked
    # for native qpos/TOPP execution; do not fabricate a Cartesian planner.
    try:
        import envs.robot.planner as planner
    except ImportError as exc:
        planner = sys.modules.get('envs.robot.planner')
        if planner is None or "CuroboPlanner" not in str(exc):
            raise
    if not hasattr(planner, 'CuroboPlanner'):
        class UnusedCartesianPlanner:
            def __init__(self, *a, **kw):
                raise RuntimeError('CuRobo unavailable; Cartesian actions unsupported')
        planner.CuroboPlanner = UnusedCartesianPlanner
    from envs.robot.robot import Robot
    from envs.handover_block import handover_block
    import yaml
    import imageio.v2 as imageio
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    from scripts.step5_arx.adapter import native_observation
    from scripts.step5_arx.data import PROMPT

    class NativeQposRobot(Robot):
        def set_planner(self, scene=None):
            self.communication_flag = False
            raw = getattr(scene, 'actual', scene)
            for side in ('left', 'right'):
                p = planner.MplibPlanner(
                    getattr(self, side + '_urdf_path'), getattr(self, side + '_srdf_path'),
                    getattr(self, side + '_move_group'), getattr(self, side + '_entity_origion_pose'),
                    getattr(self, side + '_entity'), getattr(self, side + '_planner_type'), raw)
                setattr(self, side + '_planner', p)
                setattr(self, side + '_mplib_planner', p)

    import envs._base_task as base
    base.Robot = NativeQposRobot

    settings = yaml.safe_load((pace / 'task_config/demo_clean.yml').read_text())
    camcfg = yaml.safe_load((pace / 'task_config/_camera_config.yml').read_text())
    asset = pace / 'assets/embodiments/ARX-X5'
    config = yaml.safe_load((asset / 'config.yml').read_text())
    camera = camcfg[settings['camera']['head_camera_type']]
    settings.update(task_name='handover_block', task_config='demo_clean',
                    embodiment=['ARX-X5', 'ARX-X5'], dual_arm_embodied=False,
                    embodiment_dis=0.6, left_robot_file=str(asset), right_robot_file=str(asset),
                    left_embodiment_config=config, right_embodiment_config=deepcopy(config),
                    head_camera_h=camera['h'], head_camera_w=camera['w'],
                    eval_mode=True, save_data=False, collect_data=False,
                    render_freq=0, eval_video_log=False)
    settings['data_type'] = dict(settings['data_type'], rgb=True, qpos=True, endpose=False)
    settings.pop('eval_video_save_dir', None)
    env, video, trace = handover_block(), None, None
    # Keep the actual sapien.Scene object: upstream actor factories use an
    # isinstance(Scene) check, so replacing it with a proxy breaks native setup.
    # This process-local step hook adds passive compensation and read-only logs.
    import sapien
    from types import SimpleNamespace
    clock = SimpleNamespace(tick=0, observer=None)
    original_step = sapien.Scene.step
    def physics_step(scene, *a, **kw):
        ours = scene is getattr(env, 'scene', None)
        robot = getattr(env, 'robot', None)
        if ours and robot is not None:
            for entity in (robot.left_entity, robot.right_entity):
                entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        value = original_step(scene, *a, **kw)
        if ours:
            clock.tick += 1
            if clock.observer is not None:
                clock.observer()
        return value
    sapien.Scene.step = physics_step
    result = dict(seed=args.seed, status='initializing', native_success=False,
                  execution=f'unmodified RoboTwin take_action(qpos), native TOPP, K{chunk}/H50',
                  gravity_compensation='all physics steps including initialization',
                  max_lift_mm=0., max_bilateral_lift_mm=0., grasped=False,
                  transferred_to_right=False, placed=False, actions=0, queries=0)
    write(out / 'progress.json', result)
    try:
        random.seed(args.seed)
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **settings)
        env.step_lim = min(env.step_lim,600)
        env.set_instruction(PROMPT)
        block = getattr(env.box, 'actor', env.box)
        target = getattr(env.target_box, 'actor', env.target_box)
        initial_z = float(block.get_pose().p[2])
        step0 = clock.tick
        dt = env.scene.get_timestep()
        fingers = [[j[0].child_link.entity.per_scene_id for j in getattr(env.robot, side + '_gripper')]
                   for side in ('left', 'right')]
        robot_links = {l.entity.per_scene_id: [side, l.name]
                       for side in ('left', 'right')
                       for l in getattr(env.robot, side + '_entity').get_links()}
        source_files = [pace / 'envs/_base_task.py', pace / 'envs/handover_block.py',
                        pace / 'envs/robot/robot.py', pace / 'envs/robot/planner.py',
                        asset / 'config.yml', asset / 'X5A.urdf']
        write(out / 'setup.json', dict(settings=settings, initial_block_pose=np.r_[block.get_pose().p, block.get_pose().q],
              initial_drive=np.array(env.robot.get_left_arm_jointState()+env.robot.get_right_arm_jointState()),
              initial_qpos=[e.get_qpos() for e in (env.robot.left_entity, env.robot.right_entity)],
              dt=dt, native_step_limit=env.step_lim, finger_ids=fingers,
              gripper_scale=config['gripper_scale'], native_drive='Robot.set_gripper; no force override',
              source_sha256={str(p.relative_to(pace)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}))
        dense_video = False
        result['video_sampling'] = 'every_20_physics_ticks_12.5fps' if dense_video else 'every_action_10fps'
        video = imageio.get_writer(str(out / 'head.mp4'), fps=12.5 if dense_video else 10, codec='libx264', quality=7,
                                   macro_block_size=1)
        obs = env.get_obs()
        video.append_data(obs['observation']['head_camera']['rgb'])
        trace = (out / 'physics.jsonl').open('w', buffering=1)
        current = dict(action_index=None, query_index=None)
        bilateral_start = [None, None]
        longest = [0., 0.]
        left_held = False
        right_solo = None
        def sample():
            nonlocal left_held, right_solo
            tick = clock.tick - step0
            active = [set(), set()]
            contacts = []
            support = False
            for c in env.scene.get_contacts():
                ids = [b.entity.per_scene_id for b in c.bodies]
                if block.per_scene_id not in ids:
                    continue
                impulse = max((float(np.linalg.norm(p.impulse)) for p in c.points), default=0.)
                if impulse <= 1e-8:
                    continue
                other = ids[1-ids.index(block.per_scene_id)]
                role = robot_links.get(other)
                if role is None:
                    support = True
                contacts.append(dict(ids=ids, role=role, impulse=impulse))
                for si, names in enumerate(fingers):
                    if other in names:
                        active[si].add(other)
            bilateral = [len(s) == 2 for s in active]
            z = float(block.get_pose().p[2])
            lift = (z-initial_z)*1000
            result['max_lift_mm'] = max(result['max_lift_mm'], lift)
            if any(bilateral):
                result['max_bilateral_lift_mm'] = max(result['max_bilateral_lift_mm'], lift)
            for si in (0,1):
                if bilateral[si]:
                    if bilateral_start[si] is None:
                        bilateral_start[si] = tick
                    duration = (tick-bilateral_start[si]+1)*dt
                    longest[si] = max(longest[si], duration)
                    if duration >= .2 and lift >= 30 and not support:
                        result['grasped'] = True
                        if si == 0:
                            left_held = True
                else:
                    bilateral_start[si] = None
            any_left_contact = any(c['role'] is not None and c['role'][0] == 'left' for c in contacts)
            if left_held and bilateral[1] and not any_left_contact and not support:
                if right_solo is None:
                    right_solo = tick
                if (tick-right_solo+1)*dt >= .2:
                    result['transferred_to_right'] = True
            else:
                right_solo = None
            native = bool(env.check_success())
            result['native_success'] |= native
            fp = env.box.get_functional_point(0, 'pose').p
            tp = env.target_box.get_functional_point(1, 'pose').p
            result['placed'] |= bool(native and not any(active) and support)
            trace.write(json.dumps(dict(tick=tick, sim_s=tick*dt, **current,
                  block_pose=np.r_[block.get_pose().p,block.get_pose().q], lift_mm=lift,
                  qpos=[e.get_qpos() for e in (env.robot.left_entity,env.robot.right_entity)],
                  drive=env.robot.get_left_arm_jointState()+env.robot.get_right_arm_jointState(),
                  bilateral=bilateral, contacts=contacts, support_contact=support,
                  native_success=native, box_functional_point=fp, target_functional_point=tp),default=serial)+'\n')
            if dense_video and tick % 20 == 0:
                video.append_data(env.get_obs()['observation']['head_camera']['rgb'])
        clock.observer = sample
        sample()
        with connect('ws://127.0.0.1:'+str(args.port), compression=None, max_size=32*1024**2,
                     proxy=None, ping_interval=None, open_timeout=30) as socket:
            metadata = msgpack_numpy.unpackb(socket.recv(timeout=30))
            write(out / 'model.json', metadata)
            predictions = out / 'predictions'
            predictions.mkdir()
            while env.take_action_cnt < env.step_lim and not env.eval_success:
                q = result['queries']
                obs = env.get_obs()
                request = dict(native_observation(obs), query_index=q)
                socket.send(msgpack_numpy.packb(request))
                reply = socket.recv(timeout=300)
                if isinstance(reply, str):
                    raise RuntimeError(reply)
                actions = np.asarray(msgpack_numpy.unpackb(reply)['actions'])
                if actions.shape != (50,14) or not np.isfinite(actions).all():
                    raise ValueError('expected finite H50x14 actions')
                np.savez_compressed(predictions / f'query-{q:03d}.npz',
                                    state=request['state'], actions=actions,
                                    **{cam:rgb for cam,rgb in request['images'].items()})
                result['queries'] += 1
                current['query_index'] = q
                for row in actions[:chunk]:
                    if env.take_action_cnt >= env.step_lim or env.eval_success:
                        break
                    current['action_index'] = env.take_action_cnt
                    env.take_action(row.copy(), action_type='qpos')
                    result['actions'] = env.take_action_cnt
                    # Like RoboTwin's eval video, one frame per policy action.
                    # Physical timestamps remain in physics.jsonl; video is not
                    # presented as real-time playback of the variable TOPP dt.
                    if not dense_video:
                        video.append_data(env.get_obs()['observation']['head_camera']['rgb'])
                result.update(status='running', physics_ticks=clock.tick-step0,
                              longest_bilateral_s=longest)
                write(out / 'progress.json', result)
        result.update(status='finished', stop_reason='official_success' if env.eval_success else 'native_step_limit',
                      native_success=bool(result['native_success'] or env.check_success()),
                      native_eval_success=bool(env.eval_success), final_block_z_m=float(block.get_pose().p[2]),
                      longest_bilateral_s=longest, video='head.mp4')
    except Exception:
        result.update(status='error', stop_reason='exception', traceback=traceback.format_exc())
        raise
    finally:
        if video is not None:
            video.close()
        if trace is not None:
            trace.close()
        write(out / 'complete.json', result)
        write(out / 'progress.json', result)
        if getattr(env, 'scene', None) is not None:
            env.close_env(clear_cache=True)
        sapien.Scene.step = original_step


if __name__ == '__main__':
    main()
