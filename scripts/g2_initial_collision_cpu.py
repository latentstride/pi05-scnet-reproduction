#!/usr/bin/env python3
"""Read actual settled q/fingers and audit FCL separately from PhysX contact envelopes."""
import argparse,json
from pathlib import Path
import numpy as np
import g2_settle_expert_cpu as settle
from g2_collision import JointCollisionWorld,audited_arx_self_profile

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('assets','audit','config','input','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();original=settle.capture_initial;reports=[]
    def capture(env,arms,*a,**kw):
        initial=original(env,arms,*a,**kw)
        robot='aloha-agilex' if arms[0].chain.names[0].startswith('fl_') else 'ARX-X5'
        entities=[env.robot.left_entity,env.robot.right_entity]
        checker=JointCollisionWorld.from_scene(env.scene,entities,[arm.chain.names for arm in arms],[arm.grip.names for arm in arms])
        q=np.array([initial['actual_arm_qpos_rad'][side] for side in ('left','right')])
        fingers=[]
        for entity,arm in zip(entities,arms):
            names=[j.name for j in entity.get_active_joints()];values=np.asarray(entity.get_qpos())
            fingers.append([float(values[names.index(n)]) for n in arm.grip.names])
        # Request penetration results using the same audited native ARX self policy.
        if robot=='ARX-X5':checker.use_self_profile(audited_arx_self_profile(args.assets))
        passed=checker(q,np.array(fingers))
        row=dict(episode=len(reports)//2,robot=robot,actual_arm_qpos_rad=q.tolist(),actual_native_fingers_m=fingers,
                 fcl_clear=passed,raw_collisions=checker.last_raw_collisions,
                 ignored_collisions=checker.last_ignored_collisions,blocked_collisions=checker.last_collisions)
        reports.append(row);(args.output/'initial-collision.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('FCL_INITIAL',row['episode'],robot,'clear',passed,'raw',len(row['raw_collisions']),'blocked',len(row['blocked_collisions']),flush=True)
        return initial
    settle.capture_initial=capture
    settle.run(args.assets,args.audit,args.config,args.input,args.output)
