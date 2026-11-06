#!/usr/bin/env python3
"""CPU PhysX adversarial overlaps: native internal exclusions retain cross-arm/environment contacts."""
import argparse,json
from pathlib import Path
import numpy as np
from g2_proxy_collision import collision_only_urdf
from g2_collision import apply_audited_arx_physics, physics_pair_enabled

def run(assets,output):
    import sapien
    output.mkdir(parents=True,exist_ok=False)
    path=collision_only_urdf(assets/'embodiments/ARX-X5/X5A.urdf','7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423',output/'ARX.urdf')
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.004)
    arts=[];ids=[]
    for i in range(2):
        loader=scene.create_urdf_loader();loader.fix_root_link=True
        arm=loader.load(str(path));arm.set_root_pose(sapien.Pose([0,0,1]));arm.set_name('adversarial_'+str(i))
        names=[j.name for j in arm.get_active_joints()];q=np.zeros(arm.dof)
        for n in ('joint7','joint8'):q[names.index(n)]=.044
        arm.set_qpos(q)
        for j,v in zip(arm.get_active_joints(),q):
            j.set_drive_property(stiffness=1000,damping=200,force_limit=100,mode='force')
            j.set_drive_target(float(v));j.set_drive_velocity_target(0.)
        arts.append(arm);ids.append({link.entity.per_scene_id for link in arm.get_links()})
    b=scene.create_actor_builder();b.add_box_collision(half_size=[.08,.1,.1])
    obstacle=b.build_static(name='adversarial_obstacle');obstacle.set_pose(sapien.Pose([.2,0,1.15]))
    groups_before=[list(s.get_collision_groups()) for c in obstacle.components for s in getattr(c,'collision_shapes',[])]
    profile=apply_audited_arx_physics(scene,arts,assets)
    groups_after=[list(s.get_collision_groups()) for c in obstacle.components for s in getattr(c,'collision_shapes',[])]
    cross=[];environment=[]
    for tick in range(25):
        for arm in arts:arm.set_qf(arm.compute_passive_force(gravity=True,coriolis_and_centrifugal=True))
        scene.step()
        for contact in scene.get_contacts():
            impulse=sum(float(np.linalg.norm(p.impulse)) for p in contact.points)
            if impulse<1e-8:continue
            body=[b.entity.per_scene_id for b in contact.bodies]
            item=dict(tick=tick+1,bodies=body,names=[b.entity.name for b in contact.bodies],impulse_ns=impulse)
            if (body[0] in ids[0] and body[1] in ids[1]) or (body[1] in ids[0] and body[0] in ids[1]):cross.append(item)
            if obstacle.per_scene_id in body:environment.append(item)
    get=lambda arm,name:arm.find_link_by_name(name).collision_shapes[0].get_collision_groups()
    checks=dict(cross_arm_physical_contacts=bool(cross),environment_physical_contacts=bool(environment),
                environment_filters_unchanged=groups_before==groups_after,
                native_base_link2_filtered=not physics_pair_enabled(get(arts[0],'base_link'),get(arts[0],'link2')),
                unlisted_link2_link5_enabled=physics_pair_enabled(get(arts[0],'link2'),get(arts[0],'link5')))
    r=dict(data_kind='diagnostic',gpu_used=False,renderer_created=False,physics_steps=25,scope='deliberate overlapping robot/obstacle diagnostic, not a valid task reset',
           checks=checks,passed=all(checks.values()),profile=profile,cross_examples=cross[:8],environment_examples=environment[:8])
    (output/'filter-acceptance.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return r['passed']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(0 if run(a.assets,a.output) else 1)
