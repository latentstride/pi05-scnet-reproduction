#!/usr/bin/env python3
"""Recreate expert object settling in two independent CPU robot scenes; diagnostic."""
import argparse
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from g2_assets import load_arms
from g2_adapter import Stamp
from g2_capture import capture_arms
from g2_initial import apply_fixture, capture_initial, compare_objects, digest_json
from g2_collision import apply_audited_arx_physics
from g2_proxy_collision import collision_only_urdf

def run(assets, audit_path, config_path, input_path, output, block_mass_kg=.01):
    import sapien
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(config_path.read_text());audit=json.loads(audit_path.read_text())
    candidates=json.loads(input_path.read_text());records=[]
    for entry in candidates:
        pair={}
        for robot in ('aloha-agilex','ARX-X5'):
            folder=output/(str(entry['episode'])+'-'+robot);folder.mkdir()
            scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.004)
            scene.default_physical_material=scene.create_physical_material(.5,.5,0.)
            scene.add_ground(0,render=False)
            # Match create_table: 10cm top, 10cm legs, top surface z=.74.
            builder=scene.create_actor_builder()
            builder.add_box_collision(pose=sapien.Pose([0,0,-.05]),half_size=[.6,.35,.05],material=scene.default_physical_material)
            for x in (-.55,.55):
                for y in (-.30,.30):
                    builder.add_box_collision(pose=sapien.Pose([x,y,-.372]),half_size=[.05,.05,.368])
            table=builder.build_static(name='table');table.set_pose(sapien.Pose([0,0,.74]))
            builder=scene.create_actor_builder();builder.add_box_collision(half_size=[3,.6,1.5],material=scene.default_physical_material)
            wall=builder.build_static(name='wall');wall.set_pose(sapien.Pose([0,1,1.5]))
            entities=[];asset,=[a for a in audit['robots'] if a['robot']==robot]
            urdf=collision_only_urdf(assets/asset['urdf_member'],asset['urdf_sha256'],folder/'robot.urdf')
            for index in range(1 if robot=='aloha-agilex' else 2):
                loader=scene.create_urdf_loader();loader.fix_root_link=True
                entity=loader.load(str(urdf));entity.set_name(robot+str(index));entities.append(entity)
            unique=list(entities)
            if len(entities)==1:entities=entities*2
            env=NS(robot=NS(left_entity=entities[0],right_entity=entities[1]),scene=scene,table=table)
            arms=load_arms(assets,audit_path,robot,cfg['calibration'][robot])
            samples=capture_arms(env.robot,Stamp('settle',0,0.))
            arms=tuple(replace(a,instance=s.instance) for a,s in zip(arms,samples))
            # Initialize both arms before spawning objects. Source does a 300-step open.
            for side,arm,entity in zip(('left','right'),arms,entities):
                base=cfg['reset_fixture']['robots'][robot][side]['base_world'];entity.set_root_pose(sapien.Pose(base[:3],base[3:]));names=[j.name for j in entity.get_active_joints()];q=np.array(entity.get_qpos())
                for n,v in zip(arm.chain.names+arm.grip.names,[0.]*6+arm.grip.opened.tolist()):q[names.index(n)]=v
                entity.set_qpos(q)
            for entity in unique:
                for link in entity.get_links():link.set_mass(1.)
                for j,v in zip(entity.get_active_joints(),entity.get_qpos()):
                    j.set_drive_property(stiffness=1000,damping=200,mode='force');j.set_drive_target(float(v));j.set_drive_velocity_target(0.)
            physical_filter=apply_audited_arx_physics(scene,unique,assets) if robot=='ARX-X5' else None
            for _ in range(300):scene.step()
            objects={}
            for name,half,dynamic in [('box',[.03,.03,.1],True),('target_box',[.05,.05,.005],False)]:
                ent=sapien.Entity();ent.name=name
                body=sapien.physx.PhysxRigidDynamicComponent() if dynamic else sapien.physx.PhysxRigidStaticComponent()
                body.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half,material=scene.default_physical_material))
                ent.add_component(body);pose=entry['spawn_pose'][name]
                ent.set_pose(sapien.Pose(pose['position_m'],pose['quaternion_wxyz']));scene.add_entity(ent)
                if dynamic:body.mass=block_mass_kg
                objects[name]=ent;setattr(env,name,ent)
            trajectory=[];contacts=[]
            for tick in range(2500):
                scene.step()
                if tick>=2300:
                    trajectory.append(np.r_[env.box.pose.p,env.box.pose.q])
                if tick==2499:
                    for c in scene.get_contacts():
                        contacts.append(dict(actors=[b.entity.name for b in c.bodies],points=[dict(position=p.position.tolist(),separation=float(p.separation),impulse=p.impulse.tolist()) for p in c.points]))
            traj=np.array(trajectory);body=env.box.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            initial=capture_initial(env,arms,np.eye(4),'expert-'+str(entry['episode']),digest_json(entry),'settled_cpu_reconstruction_not_formal',False)
            state=dict(initial=initial,physics_steps_after_spawn=2500,pre_spawn_steps=300,
                       last_200_position_range_m=np.ptp(traj[:,:3],axis=0).tolist(),
                       block_velocity_m_s=body.linear_velocity.tolist(),block_angular_velocity_rad_s=body.angular_velocity.tolist(),
                       block_mass_kg=float(body.mass),contacts=contacts,physical_filter=physical_filter)
            (folder/'snapshot.json').write_text(json.dumps(state,indent=2)+'\n');np.save(folder/'settle_tail.npy',traj)
            pair[robot]=state
            # Explicit scene destruction: do not leave old scenes to simulate in a global PhysX context.
            del scene,env,entities,unique,objects,body,arms
        mismatch=compare_objects(pair['aloha-agilex']['initial'],pair['ARX-X5']['initial'])
        records.append(dict(episode=entry['episode'],seed=entry['seed'],object_mismatches=mismatch,snapshots=pair))
        print('PAIR',entry['episode'],'mismatches',mismatch,flush=True)
    result=dict(data_kind='diagnostic',scope='CPU reconstructed task objects with real robot meshes; no images or policy; not original archived snapshot',
                gpu_used=False,renderer_created=False,formal_pair_ids_frozen=False,sapien_version=version('sapien'),
                block_mass_kg=block_mass_kg,block_mass_source='Actor wrapper default .01kg, not primitive shape density',
                input_sha256=sha256(input_path.read_bytes()).hexdigest(),config_sha256=sha256(config_path.read_bytes()).hexdigest(),
                asset_audit_sha256=sha256(audit_path.read_bytes()).hexdigest(),records=records)
    (output/'settling.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('assets','audit','config','input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--block-mass-kg',type=float,default=.01)
    a=p.parse_args();run(a.assets,a.audit,a.config,a.input,a.output,a.block_mass_kg)
