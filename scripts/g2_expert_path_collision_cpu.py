#!/usr/bin/env python3
"""Read actual settled q/fingers and audit FCL separately from PhysX contact envelopes."""
import argparse,json
from contextlib import nullcontext
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
        if robot=='ARX-X5':
            from mplib.sapien_utils.conversion import convert_object_name
            archive=np.load(Path(__file__).with_name('workspace-q.npz'),allow_pickle=False)
            ep=row['episode'];targets=archive['q'+str(ep)];grips=archive['g'+str(ep)]
            if targets.shape!=(len(grips),2,6) or not np.isfinite(targets).all():raise ValueError('invalid workspace data')
            from mplib.collision_detection.fcl import CollisionRequest
            class DiagnosticContact:
                request=CollisionRequest(num_max_contacts=128,enable_contact=True)
                def object_candidate(self,q):return nullcontext()
                def held_environment_collisions(self):return []
                def allows(self,row):return False
                def depth(self,result):
                    contacts=result.get_contacts()
                    if not contacts or len(contacts)>=128:return None
                    values=np.array([c.penetration_depth for c in contacts])
                    return float(max(0.,values.max())) if np.isfinite(values).all() else None
            diagnostic=DiagnosticContact()
            box_name=convert_object_name(env.box)
            prev_q=q.copy();prev_f=np.array(fingers);segments=[];examples=[];sample_count=0
            before=[np.asarray(e.get_qpos()).copy() for e in entities]
            for frame,(goal,normalized) in enumerate(zip(targets,grips)):
                fg=np.array([a.grip.target(float(v)) for a,v in zip(arms,normalized)])
                count=max(1,int(np.ceil(max(np.max(abs(goal-prev_q))/.02,np.max(abs(fg-prev_f))/.001))))
                blocked=task_unknown=0;types={}
                for fraction in np.linspace(0.,1.,count+1):
                    sample_q=prev_q+(goal-prev_q)*fraction;sample_f=prev_f+(fg-prev_f)*fraction
                    checker.evaluate(sample_q,sample_f,diagnostic);sample_count+=1
                    known=[c for c in checker.last_collisions if box_name not in (c['object_name1'],c['object_name2'])]
                    unknown=[c for c in checker.last_collisions if box_name in (c['object_name1'],c['object_name2'])]
                    blocked+=bool(known);task_unknown+=bool(unknown)
                    for c in known:
                        key=c['collision_type']+':'+c['link_name1']+':'+c['link_name2'];types[key]=types.get(key,0)+1
                    if known and len(examples)<8:examples.append(dict(frame=frame,fraction=float(fraction),q=sample_q.tolist(),collisions=known))
                segments.append(dict(frame=frame,samples=count+1,known_blocked_samples=blocked,static_box_overlap_samples=task_unknown,known_pair_counts=types))
                prev_q=goal;prev_f=fg
            unchanged=all(np.array_equal(e.get_qpos(),old) for e,old in zip(entities,before))
            result=dict(episode=ep,scope='selected IK branch, linearly sampled joint paths; held-object pose unknown and static box overlaps unresolved',
                        no_physics_mutation=unchanged,frames=len(targets),samples=sample_count,
                        segments_with_known_collision=sum(s['known_blocked_samples']>0 for s in segments),
                        max_joint_sampling_step_rad=.02,max_finger_sampling_step_m=.001,
                        full_contact_task_clear=False,segments=segments,collision_examples=examples)
            (args.output/('workspace-collision-episode'+str(ep)+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            print('FCL_PATH',ep,'samples',sample_count,'blocked_segments',result['segments_with_known_collision'],flush=True)
        return initial
    settle.capture_initial=capture
    settle.run(args.assets,args.audit,args.config,args.input,args.output)
