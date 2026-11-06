#!/usr/bin/env python3
"""Real ARX meshes + CPU physics, synthetic camera stub, fixed block; no rendering."""
import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from g2_adapter import Stamp, Snapshot, deadline, read_snapshot, solve_pair
from g2_assets import load_arms
from g2_capture import capture_arms
from g2_collision import JointCollisionWorld, audited_arx_self_profile
from g2_contact import ContactLimits, ContactTracker, ContactWorld
from g2_episode import save_json
from g2_motion import QuinticMotion
from g2_proxy_collision import collision_only_urdf
from g2_robotwin_backend import ARXBackend, pose7
from robotwin_control_geometry import inverse_pose


def run(assets,audit_path,config_path,output,dynamic=False,force=2.):
    import sapien
    from mplib.sapien_utils.conversion import convert_object_name
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads(config_path.read_text())
    audit=json.loads(audit_path.read_text())
    row,=[r for r in audit['robots'] if r['robot']=='ARX-X5']
    path=collision_only_urdf(assets/row['urdf_member'],row['urdf_sha256'],output/'ARX-collision.urdf')
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.004)
    arms=load_arms(assets,audit_path,'ARX-X5',config['calibration']['ARX-X5'])
    entities=[]
    for i in range(2):
        loader=scene.create_urdf_loader();loader.fix_root_link=True
        entity=loader.load(str(path));entity.set_name('cpu_contact_arx_'+str(i))
        entity.set_root_pose(sapien.Pose([0,i*1.5,1]))
        names=[j.name for j in entity.get_active_joints()]
        q=np.zeros(entity.dof)
        for name in arms[i].grip.names:q[names.index(name)]=.044
        entity.set_qpos(q)
        for joint in entity.get_active_joints():
            finger=joint.name in arms[i].grip.names
            joint.set_drive_property(stiffness=1000 if finger else 100000,damping=200 if finger else 1000,
                                     force_limit=2 if finger else 100,mode='force')
            joint.set_drive_target(.044 if finger else 0.);joint.set_drive_velocity_target(0.)
        entities.append(entity)
    robot=NS(left_entity=entities[0],right_entity=entities[1])
    stamp=Stamp('cpu-contact',0,0.)
    samples=capture_arms(robot,stamp)
    arms=tuple(replace(a,instance=s.instance) for a,s in zip(arms,samples))
    child=entities[0].find_joint_by_name('joint6').child_link
    child_matrix=child.entity.pose.to_transformation_matrix()
    vertices=[]
    for name in arms[0].grip.names:
        link=entities[0].find_joint_by_name(name).child_link
        points=[]
        for shape in link.collision_shapes:
            t=np.linalg.inv(child_matrix)@link.entity.pose.to_transformation_matrix()@shape.local_pose.to_transformation_matrix()
            v=np.asarray(shape.vertices)*np.asarray(shape.scale)
            points.extend(v@t[:3,:3].T+t[:3,3])
        vertices.append(np.asarray(points))
    lo=np.maximum(vertices[0].min(axis=0),vertices[1].min(axis=0))
    hi=np.minimum(vertices[0].max(axis=0),vertices[1].max(axis=0))
    local=np.eye(4);local[:3,3]=[hi[0]-.003,0.,(lo[2]+hi[2])/2]
    block_world=child_matrix@local
    builder=scene.create_actor_builder();builder.add_box_collision(half_size=[.03,.03,.1])
    if dynamic:
        # Match create_entity_box's physical body/shape construction without a renderer.
        block=sapien.Entity();block.name='box'
        body=sapien.physx.PhysxRigidDynamicComponent()
        material=scene.create_physical_material(.5,.5,0.)
        body.attach(sapien.physx.PhysxCollisionShapeBox(half_size=[.03,.03,.1],material=material))
        block.add_component(body);scene.add_entity(block)
    else:
        block=builder.build_static(name='box')
    value=pose7(block_world);block.set_pose(sapien.Pose(value[:3],value[3:]))
    support=None
    if dynamic:
        builder=scene.create_actor_builder()
        builder.add_box_collision(half_size=[.12,.12,.02],material=scene.create_physical_material(.5,.5,0.))
        support=builder.build_static(name='fixture_support')
        support.set_pose(sapien.Pose(block_world[:3,3]+[0,0,-.12]))
    builder=scene.create_actor_builder();builder.add_box_collision(half_size=[.008,.008,.008])
    obstacle=builder.build_static(name='held_obstacle');obstacle.set_pose(sapien.Pose([3,3,3]))
    base=JointCollisionWorld.from_scene(scene,entities,[a.chain.names for a in arms],[a.grip.names for a in arms])
    base.use_self_profile(audited_arx_self_profile(assets));base.audit_path=output/'collision-checks.jsonl'
    limits=ContactLimits()
    tracker=ContactTracker.from_scene(robot,block,arms,limits)
    frames=[s.world_base for s in samples]
    contact=ContactWorld(base,tracker,arms,frames,convert_object_name(block),
                         [convert_object_name(block),convert_object_name(obstacle)]+([convert_object_name(support)] if support else []),
                         [[e.find_joint_by_name(n).child_link.name for n in a.grip.names] for e,a in zip(entities,arms)],
                         [convert_object_name(support)] if support else [])
    obs=lambda:{'observation':{name:{'rgb':np.zeros((8,8,3),dtype=np.uint8)}
                              for name in ('head_camera','left_camera','right_camera')}}
    env=NS(robot=robot,scene=scene,box=block,get_obs=obs,check_success=lambda:False,
           close_env=lambda **kw:None,take_action_cnt=0)
    initial_q,initial_g,_,_=read_snapshot(Snapshot(stamp,samples,{},np.eye(4)),arms)
    initial=dict(robot_base_world={side:pose7(sample.world_base) for side,sample in zip(('left','right'),samples)},
                 actual_arm_qpos_rad={side:values.tolist() for side,values in zip(('left','right'),initial_q)},
                 actual_gripper={side:arm.grip.read(values) for side,arm,values in zip(('left','right'),arms,initial_g)})
    class Video:
        path=output/'no-render-video'
        def frame(self,obs):pass
        def close(self):pass
    drive=dict(stiffness=1000.,damping=200.,force_limit_n=force,source='CPU force sweep; provisional')
    backend=ARXBackend(env,arms,contact,np.eye(4),.004,initial,.02,5.,Video(),drive)
    backend.reset(dict(episode_id='cpu-contact'))
    snap,_=backend.capture();q,g,_,_=read_snapshot(snap,arms)
    intent=np.array([[0.,0.],[.044,.044]])
    raw_rejected=not base(q,intent)
    predicted,changes=contact.project_fingers(q,intent,deadline(20))
    bounded_accepted=contact(q,predicted)
    motion=QuinticMotion(.02,np.full((2,6),.5),np.ones((2,6)),np.full((2,2),.03),
                         np.full((2,2),.1),.02,.001,'CPU diagnostic bounds','real_mesh_contact',max_ticks=2000)
    logs=[];plans=[];replans=[];control_failure=None
    for _ in range(4):
        snap,_=backend.capture();q,g,_,_=read_snapshot(snap,arms)
        try:
            plan=motion.plan(q,g,[(q,intent)],contact,deadline(25));plans.append(plan)
            done=backend.execute(plan,logs.append,lambda:30.)
            replans.append(backend.take_replan())
        except Exception as error:
            control_failure=dict(code=getattr(getattr(error,'status',None),'value',type(error).__name__),
                                 detail=str(error),diagnostic=getattr(error,'diagnostic',{}))
            break
        if np.allclose(backend._drive_fingers(),intent,atol=1e-8,rtol=0) and tracker.holders:
            break
    snap,_=backend.capture();q,g,_,tools=read_snapshot(snap,arms)
    lift_failure=control_failure;lift_height=None
    if dynamic and control_failure is None:
        try:
            goals=tools.copy();goals[0,2,3]+=.05
            lifted,_=solve_pair(arms,frames,goals,q,intent,contact,deadline(20))
            plan=motion.plan(q,g,[(lifted,intent)],contact,deadline(25));plans.append(plan)
            backend.execute(plan,logs.append,lambda:30.)
            snap,_=backend.capture();q,g,_,tools=read_snapshot(snap,arms)
            lift_height=float(block.get_pose().p[2]-block_world[2,3])
        except Exception as error:
            lift_failure=dict(code=getattr(getattr(error,'status',None),'value',type(error).__name__),
                              detail=str(error),diagnostic=getattr(error,'diagnostic',{}))
            snap,_=backend.capture();q,g,_,tools=read_snapshot(snap,arms)
            lift_height=float(block.get_pose().p[2]-block_world[2,3])
    actual_before=np.array([e.get_qpos() for e in entities])
    drives=backend._drive_fingers()
    held=sorted(tracker.holders)
    # This is geometric transport prediction; the static block is never teleported in physics.
    candidate=q.copy();candidate[0,1]+=.4
    anchor=tracker.carrying.get('left',inverse_pose(tools[0])@block.get_pose().to_transformation_matrix())
    expected=contact.tools(candidate)[0]@anchor
    saved_holders=dict(tracker.holders)
    saved_measured=dict(tracker.measured_attachments)
    tracker.holders={'left':anchor}
    tracker.measured_attachments={'left':anchor}
    original_fcl=base.world.get_object(convert_object_name(block)).pose.to_transformation_matrix().copy()
    with base.candidate(candidate,g),contact.object_candidate(candidate):
        predicted_pose=base.world.get_object(convert_object_name(block)).pose.to_transformation_matrix()
        follows=bool(np.allclose(predicted_pose,expected,atol=1e-7,rtol=0))
    restored=bool(np.allclose(base.world.get_object(convert_object_name(block)).pose.to_transformation_matrix(),original_fcl,atol=1e-7,rtol=0))
    value=pose7(expected);obstacle.set_pose(sapien.Pose(value[:3],value[3:]));contact.refresh()
    with base.candidate(candidate,g),contact.object_candidate(candidate):
        held_hits=contact.held_environment_collisions()
    carried_obstacle_detected=bool(held_hits)
    obstacle.set_pose(sapien.Pose([3,3,3]));contact.refresh()
    still_rejects_deep=not contact(q,intent)
    right_root=entities[1].get_root_pose();entities[1].set_root_pose(entities[0].get_root_pose());contact.refresh()
    cross_arm_rejected=not contact(q,g)
    entities[1].set_root_pose(right_root);contact.refresh()
    actual_unchanged=bool(np.allclose(np.array([e.get_qpos() for e in entities]),actual_before,atol=0,rtol=0))
    tracker.holders=saved_holders
    tracker.measured_attachments=saved_measured
    checks=dict(raw_closed_endpoint_rejected=raw_rejected,bounded_prediction_accepted=bounded_accepted,
                policy_intent_preserved=bool(np.allclose(drives,intent,atol=1e-8,rtol=0)),
                actual_blocked_open=bool(np.all(g[0]>.02)),bilateral_measured_holder=held==['left'],
                measured_event_forced_replan=any(x is not None for x in replans),
                carried_object_follows_candidate=follows,fcl_pose_restored=restored,
                carried_object_environment_collision=carried_obstacle_detected,
                deep_finger_contact_rejected=still_rejects_deep,cross_arm_collision_rejected=cross_arm_rejected,
                geometric_queries_do_not_mutate_physics=actual_unchanged)
    if dynamic:checks['dynamic_lift_5cm']=lift_failure is None and lift_height is not None and lift_height>=.04
    save_json(output/'execution.json',logs)
    for i,plan in enumerate(plans):
        np.savez_compressed(output/f'plan-{i}.npz',**{k:v for k,v in asdict(plan).items() if v is not None})
    result=dict(data_kind='diagnostic',scope=('dynamic block CPU lift fixture' if dynamic else 'fixed block CPU contact integration')+'; synthetic camera stub; no full task/handover acceptance',
                gpu_used=False,renderer_created=False,physics_stepped=True,physics_steps=backend.tick,
                checks=checks,passed=all(checks.values()),dynamic_block=dynamic,block_mass_kg=float(body.mass) if dynamic else None,
                grip_measurement=[a.grip.measure(v) for a,v in zip(arms,g)],
                control_failure=control_failure,lift_failure=lift_failure,lift_height_m=lift_height,actual_fingers_m=g.tolist(),drive_fingers_m=drives.tolist(),
                holder_state=tracker.snapshot(),contact_events=tracker.events,replans=replans,
                contact_limits=asdict(limits),drive=drive,projection=changes)
    save_json(output/'contact-acceptance.json',result);print(json.dumps(result,indent=2))
    return result['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','audit','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dynamic-block',action='store_true');p.add_argument('--force-limit-n',type=float,default=2.)
    a=p.parse_args();raise SystemExit(0 if run(a.assets,a.audit,a.config,a.output,a.dynamic_block,a.force_limit_n) else 1)
