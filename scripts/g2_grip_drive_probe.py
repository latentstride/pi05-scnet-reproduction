#!/usr/bin/env python3
"""CPU force-limited position-drive assay; static block, locked-pose arm, no renderer."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from g2_proxy_collision import collision_only_urdf
from g2_robotwin_backend import pose7


def run(assets,output):
    import sapien
    output.mkdir(parents=True,exist_ok=False)
    original=assets/"X5A.urdf"
    expected="7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423"
    cpu_urdf=collision_only_urdf(original,expected,output/"ARX-collision.urdf")
    records=[]
    for name,block_enabled,force in (("empty_cap2",False,2.),("block_uncapped",True,None),
                                     ("block_cap0p5",True,.5),("block_cap2",True,2.)):
        scene=sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(.004)
        loader=scene.create_urdf_loader()
        loader.fix_root_link=True
        arm=loader.load(str(cpu_urdf))
        arm.set_name("probe_arx")
        arm.set_root_pose(sapien.Pose([0,0,1]))
        joints=arm.get_active_joints()
        names=[joint.get_name() for joint in joints]
        q=np.zeros(arm.get_dof(),dtype=np.float32)
        indices=[names.index("joint7"),names.index("joint8")]
        arm_indices=[names.index("joint"+str(i)) for i in range(1,7)]
        q[indices]=.044
        arm.set_qpos(q)
        for joint in joints:
            if joint.get_name() in ("joint7","joint8"):
                if force is None:
                    joint.set_drive_property(stiffness=1000,damping=200,mode="force")
                else:
                    joint.set_drive_property(stiffness=1000,damping=200,force_limit=force,mode="force")
                joint.set_drive_target(.044)
            else:
                joint.set_drive_property(stiffness=100000,damping=1000,force_limit=100,mode="force")
                joint.set_drive_target(0.)
            joint.set_drive_velocity_target(0.)
        child=arm.find_joint_by_name("joint6").child_link
        child_matrix=child.get_pose().to_transformation_matrix()
        local_vertices=[]
        for joint_name in ("joint7","joint8"):
            link=arm.find_joint_by_name(joint_name).child_link
            points=[]
            for shape in link.collision_shapes:
                if not hasattr(shape,"vertices"):
                    raise ValueError("mesh-only finger geometry expected")
                t=np.linalg.inv(child_matrix)@link.get_pose().to_transformation_matrix()@shape.local_pose.to_transformation_matrix()
                vertices=np.asarray(shape.vertices)*np.asarray(shape.scale)
                points.extend(vertices@t[:3,:3].T+t[:3,3])
            local_vertices.append(np.array(points))
        lo=np.maximum(local_vertices[0].min(axis=0),local_vertices[1].min(axis=0))
        hi=np.minimum(local_vertices[0].max(axis=0),local_vertices[1].max(axis=0))
        # Place the long block at the shared distal pad region, centered between the jaws.
        block_local=np.eye(4)
        block_local[:3,3]=[hi[0]-.003,0.,(lo[2]+hi[2])/2]
        block_world=child_matrix@block_local
        block=None
        if block_enabled:
            builder=scene.create_actor_builder()
            builder.add_box_collision(half_size=[.03,.03,.1])
            block=builder.build_static(name="probe_block")
            p=pose7(block_world)
            block.set_pose(sapien.Pose(p[:3],p[3:]))
        step_records=[]
        for tick in range(750):
            time_s=(tick+1)*.004
            # Identical 1.5 s command ramp in all conditions, then 1.5 s hold.
            target=.044*max(0.,1-time_s/1.5)
            for index in indices:
                joints[index].set_drive_target(target)
                joints[index].set_drive_velocity_target(0.)
            arm.set_qf(arm.compute_passive_force(gravity=True,coriolis_and_centrifugal=True))
            scene.step()
            contacted=set()
            normal_forces={}
            depths=[]
            for contact in scene.get_contacts():
                body_names=[body.entity.name for body in contact.bodies]
                if "probe_block" not in body_names:continue
                other=body_names[1] if body_names[0]=="probe_block" else body_names[0]
                contacted.add(other)
                impulses=[np.linalg.norm(np.asarray(point.impulse)) for point in contact.points]
                normal_forces[other]=normal_forces.get(other,0.)+sum(impulses)/.004
                depths.extend(max(0.,-float(point.separation)) for point in contact.points)
            step_records.append(dict(tick=tick+1,sim_s=time_s,drive_target_m=target,
                                     actual_fingers_m=np.asarray(arm.get_qpos())[indices].tolist(),
                                     contacts=sorted(contacted),contact_force_estimate_n=normal_forces,
                                     max_contact_penetration_m=max(depths,default=0.),
                                     arm_q_rad=np.asarray(arm.get_qpos())[arm_indices].tolist()))
        tail=step_records[-100:]
        actual=np.mean([row["actual_fingers_m"] for row in tail],axis=0)
        record=dict(condition=name,force_limit_n=force,block_is_static=block_enabled,
                    closed_drive_target_m=step_records[-1]["drive_target_m"],
                    settled_actual_fingers_m=actual.tolist(),block_world=block_world.tolist(),
                    settled_contact_links=sorted(set(link for row in tail for link in row["contacts"])),
                    settled_penetration_max_m=max(row["max_contact_penetration_m"] for row in tail),
                    settled_contact_force_mean_n={link:float(np.mean([row["contact_force_estimate_n"].get(link,0.) for row in tail]))
                                                 for link in set(link for row in tail for link in row["contacts"])},
                    arm_drift_max_rad=max(abs(x) for row in tail for x in row["arm_q_rad"]))
        records.append(record)
        (output/(name+".json")).write_text(json.dumps(step_records,indent=2)+"\n")
    result=dict(data_kind="diagnostic",gpu_used=False,renderer_created=False,physics_backend="PhysxCpuSystem",
                physics_stepped=True,steps_per_condition=750,physics_dt_s=.004,
                versions={"sapien":version("sapien")},urdf_sha256=sha256(original.read_bytes()).hexdigest(),
                scope="static-obstacle finger closure assay, not dynamic grasp/lift or complete task acceptance",
                conditions=records)
    (output/"grip-drive-results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.assets,a.output)
