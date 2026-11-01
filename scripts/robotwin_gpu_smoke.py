"""Run inside the isolated RoboTwin research image, never the LIBERO image.

Targets SAPIEN 3.0.0b1 and CuRobo 0.7.8, as pinned by RoboTwin stable_2.0.
The Slurm wrapper provides external timeouts and captures failures.
"""
import argparse
import json
from pathlib import Path
import importlib.metadata
import numpy as np


def render(out):
    import sapien
    from PIL import Image
    sapien.render.set_global_config(offscreen_only=True)
    scene = sapien.Scene()
    scene.set_timestep(1 / 100)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0, -1], [1, 1, 1])
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial()
    material.base_color = [1, 0.1, 0.1, 1]
    builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=material)
    box = builder.build_kinematic(name='red_box')
    # Camera forward is +x, left is +y, up is +z.
    camera = scene.add_camera('acceptance', 160, 120, 1.0, 0.01, 10)
    camera.set_pose(sapien.Pose([-0.8, 0, 0]))
    def picture():
        scene.step()
        scene.update_render()
        camera.take_picture()
        rgba = np.asarray(camera.get_picture('Color'))
        assert rgba.shape == (120,160,4) and np.isfinite(rgba).all()
        return (rgba[:,:,:3]*255).clip(0,255).astype(np.uint8)
    box.set_pose(sapien.Pose([0,0,0]))
    first = picture()
    box.set_pose(sapien.Pose([0,0.25,0]))
    second = picture()
    assert first.max() > first.min(), 'Uniform RGB is not acceptance'
    changed = int(np.count_nonzero(np.any(first != second, axis=-1)))
    assert changed > 100, 'Image must respond to moving geometry'
    Image.fromarray(first).save(out/'sapien_rgb.png')
    Image.fromarray(second).save(out/'sapien_rgb_moved.png')
    return {'pass':True,'changed_pixels':changed,'shape':list(first.shape)}


def plan(out):
    import torch
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    assert torch.cuda.is_available()
    # Small stock-UR5e planning smoke; this does NOT validate target robot assets.
    world = {'cuboid':{'table':{'dims':[5,5,0.2],'pose':[0,0,-0.1,1,0,0,0]}}}
    cfg = MotionGenConfig.load_from_robot_config('ur5e.yml',world,interpolation_dt=0.01)
    motion = MotionGen(cfg)
    motion.warmup()
    names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
             'wrist_1_joint','wrist_2_joint','wrist_3_joint']
    start = JointState.from_position(torch.zeros(1,6,device='cuda'),joint_names=names)
    goal = Pose.from_list([-0.4,0,0.4,1,0,0,0])
    result = motion.plan_single(start,goal,MotionGenPlanConfig(max_attempts=1))
    assert bool(result.success.item()), 'CuRobo planner returned failure'
    traj = result.get_interpolated_plan().position.detach().cpu().numpy()
    assert traj.ndim == 2 and len(traj)>1 and np.isfinite(traj).all()
    np.save(out/'curobo_trajectory.npy',traj)
    return {'pass':True,'shape':list(traj.shape),'robot':'stock ur5e; not target-embodiment acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['render','plan'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result = (render if args.stage == 'render' else plan)(args.output)
    result['versions'] = {p:importlib.metadata.version(p) for p in ['sapien','torch']}
    (args.output/f'{args.stage}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
