"""Diagnose and replan only the post-handover suffix using the frozen CPU gate."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np
import sapien
from mplib.collision_detection.fcl import collide
from mplib.sapien_utils.conversion import convert_object_name

from g2_adapter import Stamp, deadline, read_snapshot, tool_ik
from g2_episode import json_value, save_json
from g2_gate_scene_cpu import build
from g2_robotwin_backend import pose7
from robotwin_control_geometry import inverse_pose

def pose(matrix):
    value = pose7(matrix)
    return sapien.Pose(value[:3], value[3:])

def restore(scene, trace, index, *, drives):
    states = trace['actual_native'][index].reshape(2, 8)
    commands = trace['drive_native'][index].reshape(2, 8)
    for entity, arm, state, command in zip(scene.entities, scene.arms, states, commands):
        joints = entity.get_active_joints()
        names = arm.chain.names + arm.grip.names
        entity.set_qpos(np.array([state[names.index(j.name)] for j in joints]))
        entity.set_qvel(np.zeros(entity.dof))
        for joint in joints:
            j = names.index(joint.name)
            joint.set_drive_target(float(command[j] if drives else state[j]))
            joint.set_drive_velocity_target(0.)
    value = trace['block_pose'][index]
    scene.block.set_pose(sapien.Pose(value[:3], value[3:]))
    body = scene.block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    body.set_linear_velocity(trace['block_velocity'][index, :3] if drives else np.zeros(3))
    body.set_angular_velocity(trace['block_velocity'][index, 3:] if drives else np.zeros(3))
    for side, arm, command in zip(('left','right'), scene.arms, commands):
        setattr(scene.robot, side+'_gripper_val', arm.grip.read(command[6:]))
    scene.contact.refresh()
    return states, commands

def fcl_fingers(scene):
    model = scene.checker.models[1].get_fcl_model()
    return [obj for obj in model.get_collision_objects() if obj.name in scene.contact.finger_links[1]]

def manifold(scene):
    target = scene.checker.world.get_object(convert_object_name(scene.target))
    block = scene.checker.world.get_object(scene.contact.block_name)
    pairs = []
    for finger in fcl_fingers(scene):
        for other in (target, block):
            result = collide(finger, other, scene.contact.request)
            pairs.append(dict(finger=finger.name, other=other.name,
                              points=[dict(position=c.pos, normal=c.normal, depth_m=c.penetration_depth)
                                      for c in result.get_contacts()]))
    return pairs

def mesh_snapshot(scene):
    result = {}
    for entity in (scene.target, scene.block):
        body = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
        result[entity.name] = dict(world=entity.pose.to_transformation_matrix(),
            shapes=[dict(kind=type(s).__name__, half_size_m=s.half_size,
                         local_pose=s.local_pose.to_transformation_matrix()) for s in body.collision_shapes])
    for name in scene.arms[1].grip.names:
        link = scene.entities[1].find_joint_by_name(name).child_link
        result[link.name] = dict(world=link.entity.pose.to_transformation_matrix(),
            shapes=[dict(kind=type(s).__name__, vertices=s.vertices, scale=s.scale,
                         triangles=s.triangles if hasattr(s,'triangles') else None,
                         local_pose=s.local_pose.to_transformation_matrix()) for s in link.collision_shapes])
    return result

def diagnosis(scene, trace, args, start):
    ticks = trace['sample_ticks']
    impulses = trace['contacts_4ms'][:,2:4].sum(axis=1)
    physical_tick = int(np.flatnonzero(impulses > 1e-8)[0])
    first = {}
    for index in range(np.searchsorted(ticks,start), np.searchsorted(ticks,physical_tick+200)):
        measured, commands = restore(scene,trace,index,drives=False)
        q, g = measured[:,:6], measured[:,6:]
        block = scene.block.pose.to_transformation_matrix()
        relative = inverse_pose(scene.contact.tools(q)[1]) @ block
        scene.tracker.holders = {'right':relative}
        scene.tracker.measured_attachments = {'right':relative}
        tick = int(ticks[index])
        if 'measured' not in first and not scene.contact.measured_clear(q,g):
            with scene.checker.candidate(q,g):
                first['measured'] = dict(tick=tick, q=q, fingers=g, blocked=scene.contact.last_collisions,
                                         fcl_manifold=manifold(scene), actual_geometry=mesh_snapshot(scene))
        if 'planned' not in first:
            projected = scene.contact.planning_geometry(commands[:,:6],commands[:,6:],g,deadline(15))
            if not scene.contact(commands[:,:6],projected):
                with scene.checker.candidate(commands[:,:6],projected),scene.contact.object_candidate(commands[:,:6]):
                    first['planned'] = dict(tick=tick, q=commands[:,:6], drive_fingers=commands[:,6:],
                        projected_fingers=projected, blocked=scene.contact.last_collisions,
                        fcl_manifold=manifold(scene), actual_geometry=mesh_snapshot(scene),
                        predicted_block_pose=scene.contact.held_pose_world(commands[:,:6]),
                        candidate_finger_poses={f.name:dict(p=f.pose.p,q=f.pose.q) for f in fcl_fingers(scene)})
        if len(first)==2:
            break
    events = [json.loads(line) for line in (args.measured/'contact-events.jsonl').open()]
    first_physical = next((e for e in events if e['tick']==physical_tick), None)
    save_json(args.output/'collision-diagnosis.json', dict(first=first, first_recorded_physx_impulse_tick=physical_tick,
        archived_contact_event=first_physical, archive_impulse_columns=trace['contacts_4ms'][physical_tick,2:4],
        normal_scope='FCL manifold normals are reconstructed for archived geometry. The original PhysX event stores impulses and separation, but no normals.',
        search_start_tick=start))
    print('DIAGNOSIS', {key:value['tick'] for key,value in first.items()},physical_tick,flush=True)

def run_suffix(scene, trace, index, heights, output, original_anchor=None):
    suffix_started = monotonic()
    measured, commands = restore(scene,trace,index,drives=True)
    tracker, backend = scene.tracker, scene.backend
    for attr in ('holders','measured_attachments','candidates'):
        getattr(tracker,attr).clear()
    tracker.releasing.clear()
    tracker.last_stamp = None
    tracker.counts = dict(left=0,right=0)
    backend.episode = 'placement-suffix'
    reference_success = backend.env.check_success
    backend.env.check_success = lambda: False  # complete release and retreat before testing the original predicate
    backend.tick = 0
    steps = []
    phases = []
    attempts = []
    contacts = {}
    plan_calls = []
    release_state = None
    def log(row):
        nonlocal release_state
        steps.append(row)
        if release_state is None and tracker.ever_held and not tracker.holders:
            block = scene.block.pose.to_transformation_matrix()
            release_state = dict(tick=backend.tick, block_pose=block,
                bottom_clearance_m=float(block[2,3] - np.abs(block[2,:3]) @ [.03,.03,.10]
                                         - scene.target.pose.p[2] - .005))
        for pair in scene.scene.get_contacts():
            names = tuple(body.entity.name for body in pair.bodies)
            if scene.target.name not in names or names in contacts:
                continue
            points = [dict(position=p.position, normal=p.normal,
                           separation_m=p.separation, impulse=p.impulse)
                      for p in pair.points if np.linalg.norm(p.impulse) > 1e-8]
            if points:
                contacts[names] = dict(tick=backend.tick, pair=names,
                    body_ids=[body.entity.per_scene_id for body in pair.bodies],points=points)
    def snapshot():
        snap,_ = backend.capture()
        return read_snapshot(snap,scene.arms)

    def plan_to(name, tool, finger, budget):
        started = monotonic()
        check = deadline(budget)
        q, g, frames, tools = snapshot()
        command = backend._drive_fingers()
        command[1] = finger
        target = q.copy()
        if tool is not None:
            target[1], _ = next(tool_ik(scene.arms[1],frames[1],tool,q[1],check))
        try:
            return scene.motion.plan(q,g,[(target,command)],scene.contact,check)
        finally:
            plan_calls.append(dict(phase=name, wall_s=monotonic()-started))

    def execute_phase(name, plan):
        save_json(output/f'{name}-plan.json',asdict(plan))
        start = monotonic()
        before = backend.tick
        calls_before = len(steps)
        try:
            backend.execute(plan,log,deadline(120))
        finally:
            phases.append(dict(phase=name,physics_steps=backend.tick-before,wall_s=monotonic()-start,
                control_calls=sum(row.get('control_call_count',0) for row in steps[calls_before:])))
            save_json(output/'phases.json',phases)

    report = dict(outcome='failed', checkpoint_tick=int(trace['sample_ticks'][index]),
                  checkpoint_scope='archived joint/drive positions, block pose and velocity; joint velocities reset; contact rebuilt from actual physics')
    try:
        # Keep the recorded grasp drive and verify contact before planning any suffix.
        for _ in range(3):
            for side,(entity,_,fingers) in enumerate(backend._joints()):
                backend._apply_forces(side,entity,fingers)
            scene.scene.step()
            backend.tick += 1
            q,g,_,_ = snapshot()
            if not scene.contact.measured_clear(q,g):
                raise RuntimeError(f'restored state failed measured gate: {scene.contact.last_collisions}')
        if set(tracker.holders) != {'right'}:
            raise RuntimeError('restored checkpoint did not establish right-hand contact')
        if original_anchor is not None:
            tracker.holders['right'] = original_anchor.copy()
            report['original_slip_anchor_preserved'] = True
        q,g,frames,tools = snapshot()
        block = scene.block.pose.to_transformation_matrix()
        target = scene.target.pose.to_transformation_matrix()
        goal_xy = target[:2,3]
        base_tool = tools[1].copy()
        approach = base_tool.copy()
        approach[:2,3] += goal_xy-block[:2,3]
        approach[2,3] += .012
        t = monotonic()
        plan = plan_to('approach',approach,commands[1,6:],60)
        report['approach_plan_s'] = monotonic()-t
        execute_phase('approach',plan)
        q,g,frames,tools = snapshot()
        block = scene.block.pose.to_transformation_matrix()
        low_z = block[2,3] - np.abs(block[2,:3])@np.array([.03,.03,.10])
        tray_top = target[2,3]+.005
        for height in heights:
            t = monotonic()
            tool = tools[1].copy()
            tool[2,3] += tray_top+height-low_z
            try:
                plan = plan_to('descent',tool,commands[1,6:],60)
                # Prove that opening can be planned at the descent endpoint before any actuation.
                opening = plan.fingers[-1].copy()
                opening[1] = .044
                old_g,old_drive = scene.contact.initial_fingers.copy(),scene.contact.initial_drive.copy()
                try:
                    scene.contact.set_start(plan.predicted_fingers[-1],plan.fingers[-1])
                    preview_started = monotonic()
                    try:
                        scene.motion.plan(plan.q[-1],plan.predicted_fingers[-1],[(plan.q[-1],opening)],scene.contact,deadline(60))
                    finally:
                        plan_calls.append(dict(phase='preview-opening',wall_s=monotonic()-preview_started))
                finally:
                    scene.contact.set_start(old_g,old_drive)
                attempts.append(dict(height_m=height, accepted=True, planning_s=monotonic()-t))
                break
            except Exception as exc:
                attempts.append(dict(height_m=height, accepted=False, planning_s=monotonic()-t,
                    reason=str(exc),diagnostic=getattr(exc,'diagnostic',{}),blocked=scene.contact.last_collisions))
        else:
            raise RuntimeError('no descent plus opening candidate passed')
        execute_phase('descent',plan)
        for i in range(5):
            if not tracker.holders and np.allclose(backend._drive_fingers()[1],.044,atol=1e-7):
                break
            execute_phase(f'release-{i}',plan_to(f'release-{i}',None,.044,60))
        if tracker.holders:
            raise RuntimeError('right grasp did not release')
        _,_,_,tools = snapshot()
        retreat = tools[1].copy()
        retreat[2,3] += .035
        execute_phase('retreat',plan_to('retreat',retreat,.044,60))
        # Hold through another 0.4 s using the same sampled gate and controller.
        for i in range(20):
            execute_phase(f'hold-{i}',plan_to(f'hold-{i}',None,.044,20))
        block = scene.block.pose.to_transformation_matrix()
        delta = (block @ [0,0,-.1,1] - target @ [0,0,.005,1])[:3]
        extent_xy = np.abs(block[:2,:3]) @ np.array([.03,.03,.10])
        footprint_inside = bool(np.all(np.abs(block[:2,3]-target[:2,3])+extent_xy < .05))
        body = scene.block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        tray_clear = all(set(pair) == {scene.block.name,scene.target.name} for pair in contacts)
        report.update(outcome='passed' if reference_success() and footprint_inside and tray_clear else 'failed',
                      task_success=reference_success(), footprint_inside=footprint_inside,
                      tray_contact_scope_passed=tray_clear,
                      block_delta_m=delta, block_velocity_m_s=body.linear_velocity,
                      angular_velocity_rad_s=body.angular_velocity, holder_state=tracker.snapshot())
    except Exception as exc:
        report.update(reason=str(exc),diagnostic=getattr(exc,'diagnostic',{}),blocked=scene.contact.last_collisions)
    finally:
        backend.hold()
        report.update(phases=phases, candidates=attempts,physics_steps=backend.tick,
                      wall_s=monotonic()-suffix_started,plan_calls=plan_calls,release_state=release_state,
                      collision_checks=scene.checker.calls, physical_tray_contacts=list(contacts.values()))
        save_json(output/'suffix-result.json',report)
        with (output/'suffix-events.jsonl').open('w') as f:
            for row in steps:
                f.write(json.dumps(row,default=json_value)+'\n')
        print('SUFFIX',report['outcome'],report.get('reason'),backend.tick,flush=True)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets','audit','config','input','reference','measured','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--heights-mm',nargs='+',type=float,default=[5,7,9,12,15])
    parser.add_argument('--checkpoint',type=int,default=27434)
    parser.add_argument('--skip-diagnosis',action='store_true')
    parser.add_argument('--production-event',type=Path)
    args=parser.parse_args()
    scene=build(args.assets,args.audit,args.config,args.input,args.reference,args.output)
    with np.load(args.measured/'physics-trace.npz',allow_pickle=False) as archive:
        trace=dict(archive)
    if not args.skip_diagnosis:
        diagnosis(scene,trace,args,args.checkpoint)
    if args.production_event is not None:
        row = json.loads(args.production_event.read_text())['live_events']['last']
        state = row['contact_state']
        if state['holders'] != ['right'] or state['releasing']:
            raise ValueError('production checkpoint must have only the right holder')
        saved = dict(
            actual_native=np.c_[row['actual_arm'],row['actual_fingers']].reshape(1,16),
            drive_native=np.c_[row['target_arm'],row['target_fingers']].reshape(1,16),
            block_pose=np.array([pose7(np.array(state['block_pose_task']))]),
            block_velocity=np.zeros((1,6)),
            sample_ticks=np.array([row['tick']]),
        )
        run_suffix(scene,saved,0,np.asarray(args.heights_mm)/1000,args.output,
                   original_anchor=np.array(state['tool_block']['right']))
        return
    index=int(np.searchsorted(trace['sample_ticks'],args.checkpoint))
    if int(trace['sample_ticks'][index])!=args.checkpoint:
        raise ValueError('choose an exact archived checkpoint')
    run_suffix(scene,trace,index,np.asarray(args.heights_mm)/1000,args.output)

if __name__=='__main__':
    main()
