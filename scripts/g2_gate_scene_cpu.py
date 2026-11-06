"""Construct the audited 10 g expert scene with CPU physics and camera stubs."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from g2_adapter import Stamp,Snapshot,read_snapshot,Adapter
from g2_assets import load_arms
from g2_capture import capture_arms
from g2_collision import JointCollisionWorld,audited_arx_self_profile,apply_audited_arx_physics
from g2_contact import ContactTracker,ContactWorld,contact_profile
from g2_finger_force import centered_finger_force
from g2_motion import QuinticMotion
from g2_proxy_collision import collision_only_urdf,ProxyCollisionWorld
from g2_robotwin_backend import ARXBackend,pose7
from robotwin_control_geometry import pose_from_wxyz

def build(assets,audit_path,config_path,input_path,reference,output):
    import sapien
    from mplib.sapien_utils.conversion import convert_object_name
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads(config_path.read_text());meta=json.loads(input_path.read_text())
    audit=json.loads(audit_path.read_text());row,=[r for r in audit['robots'] if r['robot']=='ARX-X5']
    q0=np.load(reference,allow_pickle=False)['q'+str(meta['episode'])][0]
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.004)
    material=scene.create_physical_material(.5,.5,0.);scene.default_physical_material=material
    scene.add_ground(0,render=False)
    builder=scene.create_actor_builder();builder.add_box_collision(pose=sapien.Pose([0,0,-.05]),half_size=[.6,.35,.05],material=material)
    for x in (-.55,.55):
        for y in (-.30,.30):builder.add_box_collision(pose=sapien.Pose([x,y,-.372]),half_size=[.05,.05,.368])
    table=builder.build_static(name='table');table.set_pose(sapien.Pose([0,0,.74]))
    builder=scene.create_actor_builder();builder.add_box_collision(half_size=[3,.6,1.5],material=material)
    wall=builder.build_static(name='wall');wall.set_pose(sapien.Pose([0,1,1.5]))
    urdf=collision_only_urdf(assets/row['urdf_member'],row['urdf_sha256'],output/'robot.urdf')
    entities=[];groups=[]
    for si,side in enumerate(('left','right')):
        loader=scene.create_urdf_loader();loader.fix_root_link=True;e=loader.load(str(urdf));e.set_name('arx_'+side)
        base=config['reset_fixture']['robots']['ARX-X5'][side]['base_world'];e.set_root_pose(sapien.Pose(base[:3],base[3:]))
        names=[j.name for j in e.get_active_joints()];values=np.zeros(e.dof)
        for name,value in zip(row['joint_names'][si]+['joint7','joint8'],np.r_[q0[si],[.044,.044]]):values[names.index(name)]=value
        e.set_qpos(values)
        for link in e.get_links():link.set_mass(1.)
        for joint,value in zip(e.get_active_joints(),values):
            finger=joint.name in ('joint7','joint8')
            joint.set_drive_property(stiffness=0 if finger else 1000,damping=0 if finger else 200,force_limit=0 if finger else float('inf'),mode='force')
            joint.set_drive_target(float(value));joint.set_drive_velocity_target(0.)
        entities.append(e);groups.append([names.index('joint7'),names.index('joint8')])
    physical=apply_audited_arx_physics(scene,entities,assets)
    def settle_step():
        for e,ids in zip(entities,groups):
            q=np.asarray(e.get_qpos());v=np.asarray(e.get_qvel());targets=np.array([float(j.get_drive_target()[0]) for j in e.get_active_joints()])[ids]
            effort=centered_finger_force(q[ids],v[ids],targets,np.zeros(2),1000.,200.,2.)
            qf=np.asarray(e.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)).copy();qf[ids]+=effort;e.set_qf(qf)
        scene.step()
    for _ in range(100):settle_step()
    objects={}
    for name,half,dynamic in [('box',[.03,.03,.1],True),('target_box',[.05,.05,.005],False)]:
        obj=sapien.Entity();obj.name=name
        body=sapien.physx.PhysxRigidDynamicComponent() if dynamic else sapien.physx.PhysxRigidStaticComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half,material=material));obj.add_component(body)
        p=meta['spawn_pose'][name];obj.set_pose(sapien.Pose(p['position_m'],p['quaternion_wxyz']));scene.add_entity(obj)
        if dynamic:body.mass=.01
        objects[name]=obj
    for _ in range(2500):settle_step()
    block=objects['box'];target=objects['target_box']
    # This probe is fixed in the actor list, then moved only by explicit negative tests.
    builder=scene.create_actor_builder();builder.add_box_collision(half_size=[.008]*3,material=material)
    probe=builder.build_static(name='gate_probe');probe.set_pose(sapien.Pose([3,3,3]))
    robot=NS(left_entity=entities[0],right_entity=entities[1],left_gripper_val=1.,right_gripper_val=1.)
    arms=load_arms(assets,audit_path,'ARX-X5',config['calibration']['ARX-X5'])
    samples=capture_arms(robot,Stamp('gate',0,0.));arms=tuple(replace(a,instance=s.instance) for a,s in zip(arms,samples))
    limits,drive=contact_profile(config,float(block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass))
    tracker=ContactTracker.from_scene(robot,block,arms,limits)
    checker=JointCollisionWorld.from_scene(scene,entities,[a.chain.names for a in arms],[a.grip.names for a in arms])
    checker.use_self_profile(audited_arx_self_profile(assets));checker.audit_path=output/'collision-checks.jsonl'
    names=[convert_object_name(x) for x in scene.get_all_actors()]
    supports=[convert_object_name(x) for x in (table,target)]
    contact=ContactWorld(checker,tracker,arms,[s.world_base for s in samples],convert_object_name(block),names,
        [[e.find_joint_by_name(n).child_link.name for n in a.grip.names] for e,a in zip(entities,arms)],supports)
    observation=lambda:{'observation':{name:{'rgb':np.zeros((8,8,3),np.uint8)} for name in ('head_camera','left_camera','right_camera')}}
    closed=[]
    def success():
        bottom=(block.pose.to_transformation_matrix()@np.array([0,0,-.1,1]))[:3]
        top=(target.pose.to_transformation_matrix()@np.array([0,0,.005,1]))[:3]
        d=bottom-top
        return bool(np.all(abs(d[:2])<.03) and abs(d[2])<.01 and robot.right_gripper_val>.8)
    env=NS(robot=robot,scene=scene,box=block,get_obs=observation,check_success=success,close_env=lambda **kw:closed.append(True),take_action_cnt=0)
    q,g,_,_=read_snapshot(Snapshot(Stamp('gate',0,0.),samples,{},np.eye(4)),arms)
    initial=dict(robot_base_world={s:pose7(a.world_base) for s,a in zip(('left','right'),samples)},
        actual_arm_qpos_rad={s:v.tolist() for s,v in zip(('left','right'),q)},
        actual_gripper={s:a.grip.read(v) for s,a,v in zip(('left','right'),arms,g)},block_pose=np.r_[block.pose.p,block.pose.q].tolist(),
        actual_block_mass_kg=float(block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass),synthetic_cameras=True)
    class Video:
        path=output/'no-render-video'
        def frame(self,obs):pass
        def close(self):pass
    backend=ARXBackend(env,arms,contact,np.eye(4),.004,initial,config['tracking_position_limit_m'],config['tracking_angle_limit_deg'],Video(),drive)
    limits_config=config['control_limits']
    motion=QuinticMotion(config['control_period_s'],limits_config['arm_velocity_rad_s'],limits_config['arm_acceleration_rad_s2'],
        limits_config['finger_velocity_m_s'],limits_config['finger_acceleration_m_s2'],config['collision_joint_step_rad'],config['collision_finger_step_m'],limits_config['source'],'actual_scene_CPU_FCL',max_ticks=config['max_plan_ticks'])
    source=load_arms(assets,audit_path,'aloha-agilex',config['calibration']['aloha-agilex'])
    bases=[pose_from_wxyz(v[:3],v[3:]) for v in config['source_base_world']]
    source_row,=[r for r in audit['robots'] if r['robot']=='aloha-agilex']
    proxy=ProxyCollisionWorld(scene,assets/source_row['urdf_member'],source_row['urdf_sha256'],sapien.Pose(pose7(bases[0])[:3],pose7(bases[0])[3:]),source,output/'proxy.urdf')
    source_contact=ContactWorld(proxy.checker,tracker,source,bases,convert_object_name(block),names,
        [[proxy.entity.find_joint_by_name(n).child_link.name for n in a.grip.names] for a in source],supports)
    source_contact.refresh=proxy.refresh
    adapter=Adapter(source,arms,bases,np.eye(4),config['source_proxy_seed_rad'],config['source_proxy_velocity_rad_s'],source_contact,contact,motion,grip_rule=config['grip_rule'],grip_max_clip=config['grip_max_clip'])
    original_capture=backend.capture
    def capture():proxy.refresh();contact.refresh();return original_capture()
    backend.capture=capture
    return NS(scene=scene,entities=entities,robot=robot,arms=arms,block=block,target=target,probe=probe,checker=checker,contact=contact,
        tracker=tracker,backend=backend,adapter=adapter,motion=motion,initial=initial,physical_profile=physical,closed=closed,config=config)
