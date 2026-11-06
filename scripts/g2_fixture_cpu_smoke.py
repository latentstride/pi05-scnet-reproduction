#!/usr/bin/env python3
"""CPU reset/paired-snapshot geometry with real robot meshes and explicit task primitives."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from g2_adapter import Stamp, Snapshot, deadline, read_snapshot, solve_pair
from g2_assets import load_arms
from g2_capture import capture_arms
from g2_collision import JointCollisionWorld, audited_arx_self_profile
from g2_initial import apply_fixture, capture_initial, compare_objects, digest_json
from g2_proxy_collision import collision_only_urdf
from g2_robotwin_backend import pose7
from g2_motion import QuinticMotion
from robotwin_control_geometry import inverse_pose


def run(assets, audit_path, config_path, output):
    import sapien
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    fixture = config["reset_fixture"]
    records, scenes, all_arms, snapshots, checkers = {}, {}, {}, {}, {}
    for robot in ("aloha-agilex", "ARX-X5"):
        row, = [row for row in audit["robots"] if row["robot"] == robot]
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.add_ground(0, render=False)
        wall=scene.create_actor_builder()
        wall.add_box_collision(half_size=[3,.6,1.5])
        wall_actor=wall.build_static(name="wall")
        wall_actor.set_pose(sapien.Pose([0,1,1.5]))
        cpu_urdf = collision_only_urdf(assets / row["urdf_member"], row["urdf_sha256"],
                                      output / (robot + "-collision.urdf"))
        entities = []
        for index in range(1 if robot == "aloha-agilex" else 2):
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            entity = loader.load(str(cpu_urdf))
            entity.set_name(robot + str(index))
            entities.append(entity)
        if len(entities) == 1:
            entities = entities * 2
        objects = {}
        for name, half_size, dynamic in (("box", [.03,.03,.1], True),
                                         ("target_box", [.05,.05,.005], False),
                                         ("table", [.6,.35,.025], False)):
            builder = scene.create_actor_builder()
            if name == "table":
                builder.add_box_collision(pose=sapien.Pose([0,0,-.025]),half_size=half_size)
                for i in (-1,1):
                    for j in (-1,1):
                        builder.add_box_collision(pose=sapien.Pose([i*.55,j*.30,-.372]),
                                                  half_size=[.025,.025,.368])
            else:
                builder.add_box_collision(half_size=half_size)
            objects[name] = builder.build(name=name) if dynamic else builder.build_static(name=name)
        env = NS(robot=NS(left_entity=entities[0], right_entity=entities[1]), scene=scene, **objects)
        arms = load_arms(assets, audit_path, robot, config["calibration"][robot])
        actual = capture_arms(env.robot, Stamp("cpu-fixture", 0, 0.))
        arms = tuple(replace(arm, instance=sample.instance) for arm,sample in zip(arms, actual))
        apply_fixture(env, arms, fixture, robot, sapien.Pose)
        initial = capture_initial(env, arms, np.eye(4), "cpu-fixture", digest_json(fixture),
                                  "CPU_reset_geometry_not_formal_workspace", False)
        snapshot = Snapshot(Stamp("cpu-fixture", 0, 0.), capture_arms(env.robot, Stamp("cpu-fixture", 0, 0.)), {}, np.eye(4))
        collision = JointCollisionWorld.from_scene(scene, entities, [arm.chain.names for arm in arms],
                                                    [arm.grip.names for arm in arms])
        if robot=="ARX-X5":
            collision.use_self_profile(audited_arx_self_profile(assets))
        q, g, _, _ = read_snapshot(snapshot, arms)
        clear = collision(q,g)
        records[robot] = dict(initial=initial, initial_collision_free=clear, collisions=collision.last_collisions)
        scenes[robot], all_arms[robot], snapshots[robot], checkers[robot] = env, arms, snapshot, collision
    object_mismatches = compare_objects(records["aloha-agilex"]["initial"], records["ARX-X5"]["initial"])
    transfers = {}
    for source, target in (("aloha-agilex","ARX-X5"), ("ARX-X5","aloha-agilex")):
        q, g, _, tools = read_snapshot(snapshots[source], all_arms[source])
        q0, _, frames, _ = read_snapshot(snapshots[target], all_arms[target])
        normalized = [arm.grip.read(x) for arm,x in zip(all_arms[source],g)]
        target_g = np.array([arm.grip.target(value) for arm,value in zip(all_arms[target],normalized)])
        try:
            mapped,residual = solve_pair(all_arms[target], frames, tools, q0, target_g, checkers[target], deadline(20))
            entry=dict(status="ok",q=mapped.tolist(),residual=residual)
            transfers[source+"->"+target]=entry
            limits=config["control_limits"]
            motion=QuinticMotion(config["control_period_s"],limits["arm_velocity_rad_s"],
                                  limits["arm_acceleration_rad_s2"],limits["finger_velocity_m_s"],
                                  limits["finger_acceleration_m_s2"],config["collision_joint_step_rad"],
                                  config["collision_finger_step_m"],limits["source"],"CPU_real_mesh_fixture")
            _,initial_g,_,_=read_snapshot(snapshots[target],all_arms[target])
            before=checkers[target].calls
            try:
                path=motion.plan(q0,initial_g,[(mapped,target_g)],checkers[target],deadline(20))
                np.savez_compressed(output/("connection-"+source+"-to-"+target+".npz"),
                                    q=path.q,fingers=path.fingers,period_s=path.period_s)
                entry["connection_plan"]=dict(status="ok",ticks=len(path.q)-1,
                                               collision_calls=checkers[target].calls-before)
            except Exception as error:
                entry["connection_plan"]=dict(status=getattr(getattr(error,"status",None),"value","error"),
                                               detail=str(error),diagnostic=getattr(error,"diagnostic",{}))
        except Exception as exc:
            transfers[source+"->"+target] = dict(status=getattr(getattr(exc,"status",None),"value","error"),
                                                detail=str(exc), diagnostic=getattr(exc,"diagnostic",{}))
    # The imported self-ignore policy must not remove cross-arm/environment checks.
    env=scenes["ARX-X5"]
    checker=checkers["ARX-X5"]
    q,g,_,_=read_snapshot(snapshots["ARX-X5"],all_arms["ARX-X5"])
    original_right=env.robot.right_entity.get_root_pose()
    env.robot.right_entity.set_root_pose(env.robot.left_entity.get_root_pose())
    checker.refresh()
    interarm_detected=not checker(q,g)
    interarm_details=checker.last_collisions.copy()
    env.robot.right_entity.set_root_pose(original_right)
    obstacle=env.scene.create_actor_builder()
    obstacle.add_box_collision(half_size=[2,2,2])
    blocker=obstacle.build_static(name="profile_regression_obstacle")
    blocker.set_pose(sapien.Pose([0,0,1]))
    checker=JointCollisionWorld.from_scene(env.scene,[env.robot.left_entity,env.robot.right_entity],
                                          [arm.chain.names for arm in all_arms["ARX-X5"]],
                                          [arm.grip.names for arm in all_arms["ARX-X5"]])
    checker.use_self_profile(audited_arx_self_profile(assets))
    environment_detected=not checker(q,g)
    result = dict(data_kind="diagnostic", gpu_used=False, renderer_created=False, physics_stepped=False,
                  scope="two real-mesh CPU robot fixtures plus simple task primitives; not full PACE task setup",
                  fixture_sha256=digest_json(fixture), robots=records, object_mismatches=object_mismatches,
                  transfers=transfers, formal_pair_id=None, formal_workspace_certified=False,
                  arx_self_collision_profile=audited_arx_self_profile(assets),
                  scoped_profile_interarm_collision_detected=interarm_detected,
                  scoped_profile_environment_collision_detected=environment_detected,
                  scoped_profile_interarm_collision_examples=interarm_details[:8],
                  passed=not object_mismatches and interarm_detected and environment_detected)
    # Collision/IK failure is retained as a screening result rather than silently excluded.
    (output / "fixture-acceptance.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result["passed"]


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.assets,args.audit,args.config,args.output) else 1)
