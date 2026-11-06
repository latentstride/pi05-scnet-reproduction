#!/usr/bin/env python3
"""Isolate the observed withdrawal stall in one ARX articulation, CPU only."""
import argparse,json
from pathlib import Path
import numpy as np
from g2_proxy_collision import collision_only_urdf

def run(assets,output,apply_profile=False):
    import sapien
    output.mkdir(parents=True,exist_ok=False)
    path=collision_only_urdf(assets/'embodiments/ARX-X5/X5A.urdf','7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423',output/'ARX.urdf')
    actual=[.000364933,-.230694443,.177573904,-.495476931,-.000267008,.002017774,.043999,.043984]
    target=[.000029151,-.318553842,.177800341,-.496266107,-.000241429,.002017576,.044,.044]
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.004)
    loader=scene.create_urdf_loader();loader.fix_root_link=True
    arm=loader.load(str(path));arm.set_root_pose(sapien.Pose([0,0,1]))
    joints=arm.get_active_joints();names=[j.name for j in joints]
    q=np.zeros(arm.dof)
    for i in range(8):q[names.index('joint'+str(i+1))]=actual[i]
    arm.set_qpos(q)
    for i in range(8):
        j=joints[names.index('joint'+str(i+1))]
        j.set_drive_property(stiffness=100000 if i<6 else 1000,damping=1000 if i<6 else 200,force_limit=100 if i<6 else 20,mode='force')
        j.set_drive_target(target[i]);j.set_drive_velocity_target(0.)
    groups={link.name:[list(shape.get_collision_groups()) for shape in link.collision_shapes] for link in arm.get_links()}
    profile=None
    if apply_profile:
        from g2_collision import apply_audited_arx_physics
        profile=apply_audited_arx_physics(scene,[arm],assets)
    rows=[]
    for tick in range(125):
        arm.set_qf(arm.compute_passive_force(gravity=True,coriolis_and_centrifugal=True));scene.step()
        pairs=[]
        for contact in scene.get_contacts():
            impulse=sum(float(np.linalg.norm(p.impulse)) for p in contact.points)
            if impulse>1e-8:
                pairs.append(dict(bodies=[b.entity.name for b in contact.bodies],impulse_ns=impulse,
                                  depth_m=max(max(0.,-p.separation) for p in contact.points)))
        if tick%10==0 or tick==124:rows.append(dict(tick=tick+1,q=arm.get_qpos().tolist(),contacts=pairs))
    result=dict(data_kind='diagnostic',gpu_used=False,renderer_created=False,scope='isolated ARX from measured failed state; no task actors',collision_groups_before=groups,physics_profile=profile,joint_order=names,drive_target=target,rows=rows)
    (output/'drive-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--apply-audited-profile',action='store_true');a=p.parse_args();run(a.assets,a.output,a.apply_audited_profile)
