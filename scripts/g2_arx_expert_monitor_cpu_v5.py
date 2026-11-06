#!/usr/bin/env python3
"""ARX CPU expert reference with bounded aperture/center force allocation."""
import argparse,json,time,hashlib,signal
from g2_finger_force import centered_finger_force,MODE
from g2_arx_reference import retime_reference
from g2_collision import apply_audited_arx_physics
from g2_assets import load_arms
from g2_contact import ContactTracker,ContactLimits
from g2_contact import finger_force_limits
from g2_adapter import Stamp,G2Error
from types import SimpleNamespace as NS
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

def run(assets,audit_path,input_path,trace_path,output,mass=.01,force=2.,reference=None,config_path=None):
    import sapien
    def terminate(signum,frame):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,terminate)
    force=finger_force_limits(force).tolist()
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    meta=json.loads(input_path.read_text());audit=json.loads(audit_path.read_text());cfg,=[r for r in audit['robots'] if r['robot']=='ARX-X5']
    config=json.loads(config_path.read_text());archive=np.load(reference,allow_pickle=False);ep=meta['episode']
    source=unpack_trace(trace_path)
    data,retiming=retime_reference(archive['q'+str(ep)],archive['g'+str(ep)],source['frame_ticks'])
    meta=dict(meta);segments=[]
    for seg in meta['segments']:
        indices=np.flatnonzero(source['frame_ticks']==seg['last_tick'])
        first=np.flatnonzero(source['frame_ticks']==seg['first_tick'])
        segments.append(dict(seg,first_tick=int(data['frame_ticks'][first[-1]]),last_tick=int(data['frame_ticks'][indices[-1]])))
    meta['segments']=segments;meta['physics_steps']=len(data['target'])-1;n=meta['physics_steps']
    for i,seg in enumerate(segments):data['stage'][seg['first_tick']+1:seg['last_tick']+1]=i
    if not np.isfinite(data['target']).all():raise ValueError('invalid retimed controls')
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.004)
    scene.default_physical_material=scene.create_physical_material(.5,.5,0.);scene.add_ground(0,render=False)
    builder=scene.create_actor_builder();builder.add_box_collision(pose=sapien.Pose([0,0,-.05]),half_size=[.6,.35,.05],material=scene.default_physical_material)
    for x in (-.55,.55):
        for y in (-.30,.30):builder.add_box_collision(pose=sapien.Pose([x,y,-.372]),half_size=[.05,.05,.368])
    table=builder.build_static(name='table');table.set_pose(sapien.Pose([0,0,.74]))
    builder=scene.create_actor_builder();builder.add_box_collision(half_size=[3,.6,1.5],material=scene.default_physical_material)
    wall=builder.build_static(name='wall');wall.set_pose(sapien.Pose([0,1,1.5]))
    urdf=collision_only_urdf(assets/cfg['urdf_member'],cfg['urdf_sha256'],output/'robot.urdf')
    entities=[];arm_joints=[];fingers=[];ends=[]
    for si,side in enumerate(('left','right')):
        loader=scene.create_urdf_loader();loader.fix_root_link=True;e=loader.load(str(urdf));e.set_name('arx_'+side)
        base=config['reset_fixture']['robots']['ARX-X5'][side]['base_world'];e.set_root_pose(sapien.Pose(base[:3],base[3:]))
        names_i=[j.name for j in e.get_active_joints()];initial=np.zeros(e.dof)
        for name,value in zip(cfg['joint_names'][si]+['joint7','joint8'],np.r_[data['target'][0,:6] if si==0 else data['target'][0,7:13],[.044,.044]]):initial[names_i.index(name)]=value
        e.set_qpos(initial)
        for link in e.get_links():link.set_mass(1.)
        for j,value in zip(e.get_active_joints(),initial):
            finger=j.name in ('joint7','joint8')
            j.set_drive_property(stiffness=0 if finger else 1000,damping=0 if finger else 200,force_limit=0 if finger else float('inf'),mode='force')
            j.set_drive_target(float(value));j.set_drive_velocity_target(0.)
        entities.append(e);arm_joints.append([e.find_joint_by_name(k) for k in cfg['joint_names'][si]])
        fingers.append([e.find_joint_by_name(k) for k in ('joint7','joint8')]);ends.append(e.find_joint_by_name('joint6'))
    physical_filter=apply_audited_arx_physics(scene,entities,assets)
    arms=load_arms(assets,audit_path,'ARX-X5',config['calibration']['ARX-X5'])
    class Pair:
        def get_qpos(self):return np.concatenate([e.get_qpos() for e in entities])
        def compute_passive_force(self,**kw):return [e.compute_passive_force(**kw) for e in entities]
        def set_qf(self,values):
            for e,v in zip(entities,values):e.set_qf(v)
        def get_links(self):return [v for e in entities for v in e.get_links()]
    entity=Pair();joints=[j for e in entities for j in e.get_active_joints()]
    names=[side+':'+j.name for side,e in zip(('left','right'),entities) for j in e.get_active_joints()]
    indices=[offset+[j.name for j in e.get_active_joints()].index(name) for offset,e in zip((0,entities[0].dof),entities) for name in cfg['joint_names'][0]+['joint7','joint8']]
    finger_ids=[{j.child_link.entity.per_scene_id for j in group} for group in fingers]
    tick=0;applied_force=np.zeros((2,2))
    native_finger_indices=[[names.index(j.name) for j in group] for names,group in zip([[j.name for j in e.get_active_joints()] for e in entities],fingers)]
    def apply_forces():
        for si,(e,group,ids) in enumerate(zip(entities,fingers,native_finger_indices)):
            q=np.asarray(e.get_qpos());v=np.asarray(e.get_qvel());targets=np.array([float(j.get_drive_target()[0]) for j in group])
            applied_force[si]=centered_finger_force(q[ids],v[ids],targets,np.repeat(data['finger_velocity'][min(tick,n),si],2),1000.,200.,force[si])
            qf=np.asarray(e.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)).copy();qf[ids]+=applied_force[si];e.set_qf(qf)
    normalized=np.ones(2)
    def grip(side,value,epsilon):
        normalized[side]=float(np.clip(value,0,1))
        for joint in fingers[side]:
            joint.set_drive_target(.044*normalized[side]);joint.set_drive_velocity_target(float(data['finger_velocity'][tick,side]))
    # Explicit reset at the mapped source-home pose, before objects are spawned.
    for _ in range(100):
        apply_forces();scene.step()
    objects={}
    for name,half,dynamic in [('box',[.03,.03,.1],True),('target_box',[.05,.05,.005],False)]:
        obj=sapien.Entity();obj.name=name
        body=sapien.physx.PhysxRigidDynamicComponent() if dynamic else sapien.physx.PhysxRigidStaticComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half,material=scene.default_physical_material));obj.add_component(body)
        p=meta['spawn_pose'][name];obj.set_pose(sapien.Pose(p['position_m'],p['quaternion_wxyz']));scene.add_entity(obj)
        if dynamic:body.mass=mass
        objects[name]=obj
    block=objects['box'];target=objects['target_box'];body=block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    for _ in range(2500):
        apply_forces();scene.step()
    def pose(actor):return np.r_[actor.pose.p,actor.pose.q]
    def predicate():
        bottom=(block.pose.to_transformation_matrix()@np.array([0,0,-.1,1]))[:3]
        top=(target.pose.to_transformation_matrix()@np.array([0,0,.005,1]))[:3]
        delta=bottom-top
        return dict(success=bool(np.all(abs(delta[:2])<.03) and abs(delta[2])<.01 and normalized[1]>.8),block_functional_point_m=bottom.tolist(),target_functional_point_m=top.tolist(),delta_m=delta.tolist(),right_requested_open=bool(normalized[1]>.8))
    tracker=ContactTracker.from_scene(NS(left_entity=entities[0],right_entity=entities[1]),block,arms,ContactLimits())
    monitor_failure=None;monitor_max_slip_m=0.;monitor_max_slip_deg=0.
    all_force_values=[];actual=[];drives=[];boxposes=[];boxvel=[];ee=[];contact_values=[];all_contacts=[];sample_ticks=[];requests=[];snapshots=[];last_pairs=None;events=open(output/'contact-events.jsonl','w');tick=0
    rotation=np.eye(4);rotation[:3,:3]=np.array(cfg['global_trans_matrix'])@np.array(cfg['delta_matrix'])
    def sample(step):
        nonlocal last_pairs,monitor_failure,monitor_max_slip_m,monitor_max_slip_deg
        values=np.zeros(8);relevant=[];pairs=[]
        for c in scene.get_contacts():
            actors=[b.entity.name for b in c.bodies];as_set=set(actors)
            if 'box' not in actors and 'target_box' not in actors:continue
            ids={b.entity.per_scene_id for b in c.bodies}
            finger_side=next((i for i,group in enumerate(finger_ids) if ids.intersection(group)),None)
            impulse=sum(float(np.linalg.norm(p.impulse)) for p in c.points)
            depth=max([max(0.,-float(p.separation)) for p in c.points] or [0.])
            if finger_side is not None:
                if 'box' in actors:values[finger_side]+=impulse;values[4+finger_side]+=len(c.points)
                if 'target_box' in actors:values[2+finger_side]+=impulse;values[7]=max(values[7],depth)
            if 'box' in actors:values[6]=max(values[6],depth)
            pairs.append(tuple(sorted(str(b.entity.per_scene_id)+':'+b.entity.name for b in c.bodies)))
            relevant.append(dict(actors=actors,body_ids=[b.entity.per_scene_id for b in c.bodies],points=[dict(position=p.position.tolist(),separation=float(p.separation),impulse=p.impulse.tolist()) for p in c.points]))
        tools=np.array([j.child_link.entity.pose.to_transformation_matrix()@arm.offset() for j,arm in zip(ends,arms)])
        if monitor_failure is None:
            try:
                from g2_kinematics import pose_error
                bm=block.pose.to_transformation_matrix()
                for si,side in enumerate(('left','right')):
                    if step%5==0 and side in tracker.holders and side not in tracker.releasing:
                        err=pose_error(bm,tools[si]@tracker.holders[side]);monitor_max_slip_m=max(monitor_max_slip_m,float(np.linalg.norm(err[:3])));monitor_max_slip_deg=max(monitor_max_slip_deg,float(np.rad2deg(np.linalg.norm(err[3:]))))
                tracker.observe(Stamp('arx-expert-'+str(ep),step,step*.004),scene.get_contacts(),tools,bm)
            except G2Error as exc:
                monitor_failure=dict(tick=step,stage=int(data['stage'][min(step,n)]),detail=str(exc),diagnostic=exc.diagnostic,monitor_stopped_at_first_failure=True)
        all_force_values.append(applied_force.copy());all_contacts.append(values);pair_set=sorted(set(pairs))
        if step%5==0 or step in data['frame_ticks'] or step==n+200:
            actual.append(np.asarray(entity.get_qpos())[indices].copy());requests.append(normalized.copy())
            drives.append([float(joints[k].get_drive_target()[0]) for k in indices]);boxposes.append(pose(block));boxvel.append(np.r_[body.linear_velocity,body.angular_velocity])
            ee.append(tools.copy())
            contact_values.append(values);sample_ticks.append(step)
        if pair_set!=last_pairs:
            events.write(json.dumps(dict(tick=step,stage=int(data['stage'][min(step,n)]),box_pose=pose(block).tolist(),contacts=relevant))+'\n');events.flush();last_pairs=pair_set
    def flush():
        np.savez_compressed(output/'physics-trace.npz',finger_force_4ms=np.array(all_force_values),sample_ticks=np.array(sample_ticks),contacts_4ms=np.array(all_contacts),actual_native=np.array(actual),drive_native=np.array(drives),block_pose=np.array(boxposes),block_velocity=np.array(boxvel),actual_endpose_matrix=np.array(ee),contacts=np.array(contact_values),normalized_gripper_target=np.array(requests),frame_ticks=data['frame_ticks'],joint_names=np.array([names[k] for k in indices]))
        (output/'contact-monitor.json').write_text(json.dumps(dict(failure=monitor_failure,snapshot=tracker.snapshot(),events=tracker.events,max_cumulative_slip_m=monitor_max_slip_m,max_cumulative_slip_deg=monitor_max_slip_deg,max_slip_statistics_sample_ticks=5,continuous_guard_ticks=1,mode='shadow observer; first failure retained; physics not altered'),indent=2)+'\n')
        (output/'stages.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    initial=dict(actual_all_joint_names=names,actual_all_qpos=np.asarray(entity.get_qpos()).tolist(),block_pose=pose(block).tolist(),target_pose=pose(target).tolist(),block_mass_kg=float(body.mass),robot_bases=[pose(e.get_links()[0].entity).tolist() for e in entities],gripper_force_limit_n=force,initial_configuration='mapped source home; explicit reset before spawn')
    (output/'initial.json').write_text(json.dumps(initial,indent=2)+'\n');sample(0)
    boundaries={s['last_tick']:i for i,s in enumerate(meta['segments'])};error=None
    try:
        for tick in range(1,n+1):
            if time.monotonic()-start>720:raise TimeoutError('CPU playback wall deadline')
            q=data['target'][tick];vel=data['arm_velocity'][tick]
            if monitor_failure is None:
                try:tracker.authorize_opening(np.repeat(normalized[:,None]*.044,2,axis=1),np.repeat(q[[6,13],None]*.044,2,axis=1),arms)
                except G2Error as exc:monitor_failure=dict(tick=tick,stage=int(data['stage'][tick]),detail=str(exc),diagnostic=exc.diagnostic,monitor_stopped_at_first_failure=True)
            for si,group in enumerate(arm_joints):
                for j,value,v in zip(group,q[:6] if si==0 else q[7:13],vel[si]):j.set_drive_target(float(value));j.set_drive_velocity_target(float(v))
            for si in range(2):
                if data['gripper_active'][tick,si]:grip(si,q[6 if si==0 else 13],float(data['gripper_eps'][tick,si]))
            apply_forces();scene.step();sample(tick)
            if tick in boundaries:
                si=boundaries[tick];record=dict(stage=si,tick=tick,kind=meta['segments'][si]['kind'],block_pose=pose(block).tolist(),predicate=predicate())
                snapshots.append(record);flush();print('STAGE',si,'tick',tick,'block',record['block_pose'][:3],'success',record['predicate']['success'],flush=True)
        at_end=predicate()
        # Separate diagnostic hold; never substitute its result for the original end predicate.
        for hold in range(200):apply_forces();scene.step();sample(n+hold+1)
        after_hold=predicate()
    except BaseException as exc:
        error=dict(type=type(exc).__name__,detail=str(exc));at_end=predicate();after_hold=None
        raise
    finally:
        events.close();flush()
        result=dict(data_kind='diagnostic',scope='retimed sampled expert FK/IK to ARX, direct finite-force physical reference; production planner/contact gate not exercised',gpu_used=False,renderer_created=False,model_used=False,
                    finger_force_control_mode=MODE,max_allocated_finger_force_n=np.max(np.abs(all_force_values),axis=0).tolist(),episode=meta['episode'],seed=meta['seed'],contact_monitor_passed=monitor_failure is None and error is None,contact_monitor_covered_ticks=tick,contact_monitor_failure=monitor_failure,retiming=retiming,physical_filter=physical_filter,gripper_force_limit_n=force,collision_gate_exercised=False,completed=error is None,physics_steps_replayed=tick,expected_physics_steps=n,diagnostic_hold_steps=200 if error is None else 0,
                    source_actor_default_mass_kg=.01,actual_block_mass_kg=float(body.mass),native_success_at_expert_end=at_end,native_success_after_diagnostic_hold=after_hold,
                    failure=error,wall_s=time.monotonic()-start,sapien_version=version('sapien'),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                    initialization='explicit mapped source-home configuration, 100 robot settling ticks, 2500 object settling ticks with passive-force compensation; no renderer',
                    contact_columns=['left_block_impulse_Ns','right_block_impulse_Ns','left_target_impulse_Ns','right_target_impulse_Ns','left_block_contact_points','right_block_contact_points','max_block_penetration_m','max_finger_target_penetration_m'])
        (output/'replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','audit','input','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--block-mass-kg',type=float,default=.01)
    p.add_argument('--force-limit-n',type=float,nargs=2,default=[2.,2.])
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();run(a.assets,a.audit,a.input,a.trace,a.output,a.block_mass_kg,a.force_limit_n,a.reference,a.config)
