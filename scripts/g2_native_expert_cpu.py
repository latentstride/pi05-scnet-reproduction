#!/usr/bin/env python3
"""No-render CPU playback of every cached native Aloha drive update, with measured physics."""
import argparse,json,time,hashlib
from pathlib import Path
from importlib.metadata import version
import numpy as np
from g2_proxy_collision import collision_only_urdf

def unpack_trace(path):
    z=np.load(path,allow_pickle=False);out={k:z[k] for k in z.files if not k.endswith('_delta')}
    for key,shape in [('target',(-1,14)),('arm_velocity',(-1,2,6))]:
        bits=np.ascontiguousarray(z[key+'_delta'].T).view(np.uint32)
        out[key]=np.bitwise_xor.accumulate(bits,axis=0).view(np.float32).reshape(shape).astype(float)
    out['target'][:,[6,13]]=out['gripper_norm']
    return out

def run(assets,audit_path,input_path,trace_path,output,mass=.01):
    import sapien
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    meta=json.loads(input_path.read_text());audit=json.loads(audit_path.read_text());cfg,=[r for r in audit['robots'] if r['robot']=='aloha-agilex']
    data=unpack_trace(trace_path);n=len(data['target'])-1
    if n!=meta['physics_steps'] or not np.isfinite(data['target']).all():raise ValueError('invalid dense controls')
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.004)
    scene.default_physical_material=scene.create_physical_material(.5,.5,0.);scene.add_ground(0,render=False)
    builder=scene.create_actor_builder();builder.add_box_collision(pose=sapien.Pose([0,0,-.05]),half_size=[.6,.35,.05],material=scene.default_physical_material)
    for x in (-.55,.55):
        for y in (-.30,.30):builder.add_box_collision(pose=sapien.Pose([x,y,-.372]),half_size=[.05,.05,.368])
    table=builder.build_static(name='table');table.set_pose(sapien.Pose([0,0,.74]))
    builder=scene.create_actor_builder();builder.add_box_collision(half_size=[3,.6,1.5],material=scene.default_physical_material)
    wall=builder.build_static(name='wall');wall.set_pose(sapien.Pose([0,1,1.5]))
    loader=scene.create_urdf_loader();loader.fix_root_link=True
    urdf=collision_only_urdf(assets/cfg['urdf_member'],cfg['urdf_sha256'],output/'robot.urdf')
    entity=loader.load(str(urdf));entity.set_name('native_aloha')
    base=cfg['robot_pose_wxyz'][0];entity.set_root_pose(sapien.Pose(base[:3],base[3:]))
    joints=entity.get_active_joints();names=[j.name for j in joints]
    arm_names=cfg['joint_names'];finger_names=[['fl_joint7','fl_joint8'],['fr_joint7','fr_joint8']]
    arm_joints=[[entity.find_joint_by_name(k) for k in group] for group in arm_names]
    fingers=[[entity.find_joint_by_name(k) for k in group] for group in finger_names]
    indices=[names.index(k) for a,g in zip(arm_names,finger_names) for k in a+g]
    ends=[entity.find_joint_by_name(k) for k in cfg['ee_joints']]
    for link in entity.get_links():link.set_mass(1.)
    for joint in joints:joint.set_drive_property(stiffness=1000,damping=200,mode='force')
    for group in arm_joints:
        for joint in group:joint.set_drive_target(0.)
    normalized=np.zeros(2)
    def grip(side,value,epsilon):
        normalized[side]=float(np.clip(value,0,1))
        real=float(np.clip((fingers[side][0].get_drive_target()[0]+.01)/.055,0,1))
        limited=normalized[side]
        if (limited-real>epsilon and epsilon>0) or (limited-real<epsilon and epsilon<0):limited=real+epsilon
        native=-.01+.055*limited
        for joint in fingers[side]:
            old=float(joint.get_drive_target()[0]);joint.set_drive_target(native)
            joint.set_drive_velocity_target(float(np.clip(native-old,-1,1)*.05))
    # The original camera setup advances one physical tick before home/open.
    scene.step()
    warm=np.r_[np.linspace(0.,1.,200),np.ones(100)]
    for value in warm:
        entity.set_qf(entity.compute_passive_force(gravity=True,coriolis_and_centrifugal=True))
        grip(0,value,.005);grip(1,value,.005);scene.step()
    objects={}
    for name,half,dynamic in [('box',[.03,.03,.1],True),('target_box',[.05,.05,.005],False)]:
        obj=sapien.Entity();obj.name=name
        body=sapien.physx.PhysxRigidDynamicComponent() if dynamic else sapien.physx.PhysxRigidStaticComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half,material=scene.default_physical_material));obj.add_component(body)
        p=meta['spawn_pose'][name];obj.set_pose(sapien.Pose(p['position_m'],p['quaternion_wxyz']));scene.add_entity(obj)
        if dynamic:body.mass=mass
        objects[name]=obj
    block=objects['box'];target=objects['target_box'];body=block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    for _ in range(2500):scene.step()
    def pose(actor):return np.r_[actor.pose.p,actor.pose.q]
    def predicate():
        bottom=(block.pose.to_transformation_matrix()@np.array([0,0,-.1,1]))[:3]
        top=(target.pose.to_transformation_matrix()@np.array([0,0,.005,1]))[:3]
        delta=bottom-top
        return dict(success=bool(np.all(abs(delta[:2])<.03) and abs(delta[2])<.01 and normalized[1]>.8),block_functional_point_m=bottom.tolist(),target_functional_point_m=top.tolist(),delta_m=delta.tolist(),right_requested_open=bool(normalized[1]>.8))
    actual=[];drives=[];boxposes=[];boxvel=[];ee=[];contact_values=[];requests=[];snapshots=[];last_pairs=None;events=open(output/'contact-events.jsonl','w');tick=0
    rotation=np.eye(4);rotation[:3,:3]=np.array(cfg['global_trans_matrix'])@np.array(cfg['delta_matrix'])
    def sample(step):
        nonlocal last_pairs
        actual.append(np.asarray(entity.get_qpos())[indices].copy());requests.append(normalized.copy())
        drives.append([float(joints[k].get_drive_target()[0]) for k in indices]);boxposes.append(pose(block));boxvel.append(np.r_[body.linear_velocity,body.angular_velocity])
        matrices=[j.global_pose.to_transformation_matrix()@rotation for j in ends];ee.append(matrices)
        values=np.zeros(8);relevant=[];pairs=[]
        for c in scene.get_contacts():
            actors=[b.entity.name for b in c.bodies];as_set=set(actors)
            if 'box' not in actors and 'target_box' not in actors:continue
            finger_side=next((i for i,group in enumerate([['fl_link7','fl_link8'],['fr_link7','fr_link8']]) if as_set.intersection(group)),None)
            impulse=sum(float(np.linalg.norm(p.impulse)) for p in c.points)
            depth=max([max(0.,-float(p.separation)) for p in c.points] or [0.])
            if finger_side is not None:
                if 'box' in actors:values[finger_side]+=impulse;values[4+finger_side]+=len(c.points)
                if 'target_box' in actors:values[2+finger_side]+=impulse;values[7]=max(values[7],depth)
            if 'box' in actors:values[6]=max(values[6],depth)
            pairs.append(tuple(sorted(actors)))
            relevant.append(dict(actors=actors,points=[dict(position=p.position.tolist(),separation=float(p.separation),impulse=p.impulse.tolist()) for p in c.points]))
        contact_values.append(values);pair_set=sorted(set(pairs))
        if pair_set!=last_pairs:
            events.write(json.dumps(dict(tick=step,stage=int(data['stage'][min(step,n)]),box_pose=boxposes[-1].tolist(),contacts=relevant))+'\n');events.flush();last_pairs=pair_set
    def flush():
        np.savez_compressed(output/'physics-trace.npz',actual_native=np.array(actual),drive_native=np.array(drives),block_pose=np.array(boxposes),block_velocity=np.array(boxvel),actual_endpose_matrix=np.array(ee),contacts=np.array(contact_values),normalized_gripper_target=np.array(requests),frame_ticks=data['frame_ticks'],joint_names=np.array([names[k] for k in indices]))
        (output/'stages.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    initial=dict(actual_all_joint_names=names,actual_all_qpos=np.asarray(entity.get_qpos()).tolist(),block_pose=pose(block).tolist(),target_pose=pose(target).tolist(),block_mass_kg=float(body.mass),robot_base=pose(entity.get_links()[0].entity).tolist())
    (output/'initial.json').write_text(json.dumps(initial,indent=2)+'\n');sample(0)
    boundaries={s['last_tick']:i for i,s in enumerate(meta['segments'])};error=None
    try:
        for tick in range(1,n+1):
            if time.monotonic()-start>240:raise TimeoutError('CPU playback wall deadline')
            entity.set_qf(entity.compute_passive_force(gravity=True,coriolis_and_centrifugal=True))
            q=data['target'][tick];vel=data['arm_velocity'][tick]
            for si,group in enumerate(arm_joints):
                for j,value,v in zip(group,q[:6] if si==0 else q[7:13],vel[si]):j.set_drive_target(float(value));j.set_drive_velocity_target(float(v))
            for si in range(2):
                if data['gripper_active'][tick,si]:grip(si,q[6 if si==0 else 13],float(data['gripper_eps'][tick,si]))
            scene.step();sample(tick)
            if tick in boundaries:
                si=boundaries[tick];record=dict(stage=si,tick=tick,kind=meta['segments'][si]['kind'],block_pose=pose(block).tolist(),predicate=predicate())
                snapshots.append(record);flush();print('STAGE',si,'tick',tick,'block',record['block_pose'][:3],'success',record['predicate']['success'],flush=True)
        at_end=predicate()
        # Separate diagnostic hold; never substitute its result for the original end predicate.
        for hold in range(200):scene.step();sample(n+hold+1)
        after_hold=predicate()
    except Exception as exc:
        error=dict(type=type(exc).__name__,detail=str(exc));at_end=predicate();after_hold=None
        raise
    finally:
        events.close();flush()
        result=dict(data_kind='diagnostic',scope='dense native expert drive replay in reconstructed no-render original task geometry',gpu_used=False,renderer_created=False,model_used=False,
                    episode=meta['episode'],seed=meta['seed'],completed=error is None,physics_steps_replayed=tick,expected_physics_steps=n,diagnostic_hold_steps=200 if error is None else 0,
                    source_actor_default_mass_kg=.01,actual_block_mass_kg=float(body.mass),native_success_at_expert_end=at_end,native_success_after_diagnostic_hold=after_hold,
                    failure=error,wall_s=time.monotonic()-start,sapien_version=version('sapien'),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                    initialization='source-like camera tick, 200+100 gripper ramp and passive forces, 2500 settling ticks; no renderer',
                    contact_columns=['left_block_impulse_Ns','right_block_impulse_Ns','left_target_impulse_Ns','right_target_impulse_Ns','left_block_contact_points','right_block_contact_points','max_block_penetration_m','max_finger_target_penetration_m'])
        (output/'replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','audit','input','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--block-mass-kg',type=float,default=.01)
    a=p.parse_args();run(a.assets,a.audit,a.input,a.trace,a.output,a.block_mass_kg)
