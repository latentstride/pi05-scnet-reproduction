"""Run inside the isolated RoboTwin research image, never the LIBERO image.

Targets SAPIEN 3.0.0b1 and CuRobo 0.7.8, as pinned by RoboTwin stable_2.0.
The Slurm wrapper provides external timeouts and captures failures.
"""
import argparse
import json
from pathlib import Path
import importlib.metadata
import os
import time
import numpy as np


def pci_tuple(value):
    domain, bus, rest = value.lower().split(':')
    device, function = rest.split('.')
    return tuple(int(x,16) for x in (domain,bus,device,function))


def render(out):
    import sapien
    from PIL import Image
    # Verified against installed 3.0.0b1 signatures. No viewer or window.
    # CUDA_VISIBLE_DEVICES is the allocated GPU UUID, forwarded past --cleanenv.
    device = sapien.Device('cuda:0')
    assert device.is_cuda() and device.can_render() and device.cuda_id == 0
    expected_pci = os.environ['ROBOTWIN_EXPECTED_PCI']
    assert pci_tuple(device.pci_string) == pci_tuple(expected_pci), 'Wrong SAPIEN GPU'
    renderer = sapien.render.RenderSystem(device=device)
    assert pci_tuple(renderer.device.pci_string) == pci_tuple(expected_pci)
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), renderer])
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
    return {'pass':True,'changed_pixels':changed,'shape':list(first.shape),
            'gpu_name':device.name,'pci_bus_id':device.pci_string,
            'allocated_gpu_verified':True,'renderer':'SAPIEN offscreen raster'}


def plan(out):
    import torch
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    # Small stock-UR5e planning smoke; this does NOT validate target robot assets.
    world = {'cuboid':{'table':{'dims':[5,5,0.2],'pose':[0,0,-0.1,1,0,0,0]}}}
    cfg = MotionGenConfig.load_from_robot_config(
        'ur5e.yml',world,interpolation_dt=0.01,use_cuda_graph=False)
    print('CuRobo: initialize MotionGen', flush=True)
    motion = MotionGen(cfg)
    names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
             'wrist_1_joint','wrist_2_joint','wrist_3_joint']
    # Use the stock robot's safe retract configuration and a small base rotation,
    # following the pinned v0.7.8 example rather than assuming zero is collision-free.
    retract = motion.get_retract_config().view(1,-1).clone()
    goal = JointState.from_position(retract, joint_names=names)
    state = motion.compute_kinematics(goal)
    start_q = retract.clone()
    start_q[0,0] += 0.15
    start = JointState.from_position(start_q,joint_names=names)
    # The warmup calls the Cartesian planner even with CUDA graphs disabled.
    # Use the joint-space API for this known joint goal and check endpoint FK.
    print('CuRobo: plan_single_js', flush=True)
    result = motion.plan_single_js(start,goal,MotionGenPlanConfig(
        max_attempts=1,enable_graph=False,parallel_finetune=False))
    assert bool(result.success.item()), 'CuRobo planner returned failure'
    traj = result.get_interpolated_plan().position.detach().cpu().numpy()
    assert traj.ndim == 2 and traj.shape[1] == 6 and len(traj)>1 and np.isfinite(traj).all()
    assert float(np.max(np.abs(traj[-1]-traj[0]))) > 0.01, 'Stationary path is not a motion test'
    final = motion.compute_kinematics(JointState.from_position(
        torch.as_tensor(traj[-1:],device='cuda'),joint_names=names))
    target_p = state.ee_pos_seq.reshape(-1,3)[0]
    final_p = final.ee_pos_seq.reshape(-1,3)[0]
    position_error = float(torch.linalg.norm(final_p-target_p).item())
    qa = state.ee_quat_seq.reshape(-1,4)[0]
    qb = final.ee_quat_seq.reshape(-1,4)[0]
    cosine = torch.abs(torch.dot(qa,qb)/(torch.linalg.norm(qa)*torch.linalg.norm(qb)))
    angle_error = float((2*torch.acos(cosine.clamp(0,1))).item())
    assert position_error <= 0.01 and angle_error <= 0.1, 'Plan does not reach the target'
    np.save(out/'curobo_trajectory.npy',traj)
    return {'pass':True,'shape':list(traj.shape),'goal_position_error_m':position_error,
            'goal_angle_error_rad':angle_error,'cuda_device_count':torch.cuda.device_count(),
            'cuda_gpu_name':torch.cuda.get_device_name(0),
            'robot':'stock ur5e; not target-embodiment acceptance', 'planner':'plan_single_js'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['render','plan'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    start_time = time.monotonic()
    print('START',args.stage,flush=True)
    result = (render if args.stage == 'render' else plan)(args.output)
    result['elapsed_seconds'] = time.monotonic()-start_time
    result['versions'] = {p:importlib.metadata.version(p) for p in ['sapien','torch']}
    (args.output/f'{args.stage}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
