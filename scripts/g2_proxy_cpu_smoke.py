#!/usr/bin/env python3
"""Verify that the actual task actor geometry is copied into the Aloha proxy world."""
import argparse
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from g2_assets import load_arms
from g2_proxy_collision import ProxyCollisionWorld, collision_only_urdf


def run(assets,audit_path,config_path,output):
    import sapien
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads(config_path.read_text())
    audit=json.loads(audit_path.read_text())
    source,= [r for r in audit["robots"] if r["robot"]=="aloha-agilex"]
    target,= [r for r in audit["robots"] if r["robot"]=="ARX-X5"]
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()])
    path=collision_only_urdf(assets/target["urdf_member"],target["urdf_sha256"],output/"ARX.urdf")
    entities=[]
    for i,x in enumerate((-.3,.3)):
        loader=scene.create_urdf_loader()
        loader.fix_root_link=True
        entity=loader.load(str(path))
        entity.set_name("actual_arx_"+str(i))
        entity.set_root_pose(sapien.Pose([x,-.35,.784],[.70710678,0,0,.70710678]))
        entities.append(entity)
    builder=scene.create_actor_builder()
    builder.add_box_collision(half_size=[2,2,2])
    obstacle=builder.build_static(name="copied_obstacle")
    obstacle.set_pose(sapien.Pose([10,10,10]))
    arms=load_arms(assets,audit_path,"aloha-agilex",config["calibration"]["aloha-agilex"])
    base=config["source_base_world"][0]
    proxy=ProxyCollisionWorld(scene,assets/source["urdf_member"],source["urdf_sha256"],
                              sapien.Pose(base[:3],base[3:]),arms,output/"aloha-proxy.urdf")
    q=np.zeros((2,6))
    fingers=np.array([a.grip.target(1.) for a in arms])
    before=[e.get_qpos().copy() for e in entities]
    clear_before=proxy(q,fingers)
    obstacle.set_pose(sapien.Pose([0,0,.7]))
    proxy.refresh()
    blocked=not proxy(q,fingers)
    unchanged=all(np.array_equal(old,e.get_qpos()) for old,e in zip(before,entities))
    result=dict(data_kind="diagnostic",gpu_used=False,renderer_created=False,physics_stepped=False,
                versions={"sapien":version("sapien"),"mplib":version("mplib")},
                clear_before=clear_before,copied_obstacle_detected_after_refresh=blocked,
                actual_arx_qpos_unchanged=unchanged,collisions=proxy.last_collisions,
                passed=clear_before and blocked and unchanged)
    (output/"proxy-acceptance.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result["passed"]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("assets","audit","config","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    raise SystemExit(0 if run(a.assets,a.audit,a.config,a.output) else 1)
