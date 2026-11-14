"""Real CPU regression of the production adapter and its observation boundary.

Archived-tail mode is explicitly counterfactual. Interface mode stops after
the first actual reobservation request; no substitute policy action is executed.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace as NS

import numpy as np

from g2_adapter import G2Error,deadline
from g2_arx_direct import DirectAdapter,execute_prefix
from g2_arx_endpoint_cpu import build_scene,save
from g2_arx_transfer_probe_cpu import restart_backend
from pace_runtime import validate_request
from robotwin_control_geometry import Status


def run(a):
    spec=json.loads(a.input.read_text());cfg=spec['config'];a.output.mkdir()
    scene,old,snap,drive=build_scene(spec,a.audit,a.assets,a.output)
    backend,_,_=restart_backend(scene,old,snap,drive,cfg,spec['initial'],a.assets)
    actors={v.name:v for v in scene.get_all_actors()}
    backend.env.table,backend.env.target_box=actors['table'],actors['target_box']
    backend.env.get_obs=lambda:{'observation':{k:{'rgb':np.zeros((8,8,3),dtype=np.uint8)}
        for k in ('head_camera','left_camera','right_camera')}}
    backend.env.check_success=lambda:False;backend.video=NS(frame=lambda _:None)
    adapter=DirectAdapter(old.aloha,old.arx,old.source_frames,np.eye(4),old.seed,old.source_speed,
        target_collision=old.target_collision,motion=old.motion.motion,
        grip_rule=old.grip_rule,grip_max_clip=old.grip_max_clip)
    receipt=dict(job=spec['job'],mode=a.mode,policy_source_sha256=spec['action_sha256'],
        first_row=a.first_row,start_tick=backend.tick,completed_rows=[],safe_segments=0,
        contact_steps=[0,0],max_contact_fingers=[0,0],stable_grasp=False,
        min_measured_aperture=[1.,1.],min_drive_aperture=[1.,1.],
        rgb='CPU fixture only; actual RGB/model inference GPU_REQUIRED')
    check=deadline(800);started=monotonic();remaining=lambda:check() or 800-(monotonic()-started)
    number=0
    with (a.output/'events.jsonl').open('x') as stream:
        def log(event):
            stream.write(json.dumps(dict(original_policy_row=number,**event),default=lambda v:v.tolist())+'\n')
            if event.get('physics_step_count'):
                for i,side in enumerate(('left','right')):
                    n=len((event.get('contact_state') or {}).get('contact_fingers',{}).get(side,[]))
                    receipt['max_contact_fingers'][i]=max(receipt['max_contact_fingers'][i],n)
                    if n:receipt['contact_steps'][i]+=event['physics_step_count']
                    receipt['min_measured_aperture'][i]=min(receipt['min_measured_aperture'][i],
                        adapter.arx[i].grip.read(np.array(event['actual_fingers'][i])))
                    receipt['min_drive_aperture'][i]=min(receipt['min_drive_aperture'][i],
                        adapter.arx[i].grip.read(np.array(event['target_fingers'][i])))
        try:
            snap,_=backend.capture()
            first=adapter.observe(snap,cfg['prompt'],remaining())
            if not first.ok:raise G2Error(first.status,first.detail,**first.diagnostic)
            validate_request(first.payload)
            receipt['initial_request_state']=first.payload['state'].tolist()
            rows=np.asarray(spec['chunk'])[a.first_row-1:]
            if a.mode=='negative':
                import sapien
                from mplib.sapien_utils import SapienPlanningWorld
                from sapien.physx import PhysxRigidBaseComponent
                left=next(e for e in scene.get_all_articulations() if e.name=='arx_left')
                wrist=next(link for link in left.get_links() if link.name=='link6')
                builder=scene.create_actor_builder()
                builder.add_box_collision(half_size=[.03,.03,.03])
                obstacle=builder.build_static(name='stage6_fault_probe')
                obstacle.set_pose(wrist.get_pose())
                obj=SapienPlanningWorld.convert_physx_component(obstacle.find_component_by_type(PhysxRigidBaseComponent))
                backend.collision.base.world.add_object(obj)
                before=backend.tick
                try:
                    execute_prefix(backend,snap,rows[:1],adapter.policy_row,remaining,log,lambda:None)
                except G2Error as error:
                    receipt['real_arx_collision_rejected']=error.status==Status.COLLISION and backend.tick==before
                    receipt['collision_diagnostic']=error.diagnostic
                else:receipt['real_arx_collision_rejected']=False
            elif a.mode=='interface':
                number=a.first_row
                count=[]
                prefix=execute_prefix(backend,snap,rows,adapter.policy_row,remaining,log,
                                      lambda:count.append(backend.tick))
                receipt.update(prefix=asdict(prefix),executed_policy_actions=len(count))
                fresh,_=backend.capture()
                assert fresh.stamp==prefix.stamp
                observed=adapter.observe(fresh,cfg['prompt'],remaining())
                if not observed.ok:raise G2Error(observed.status,observed.detail,**observed.diagnostic)
                validate_request(observed.payload)
                receipt['fresh_request']=dict(stamp=vars(fresh.stamp),state=observed.payload['state'].tolist(),
                    image_shapes={k:list(v.shape) for k,v in observed.payload['images'].items()},
                    objects=sorted(fresh.objects_task),validated=True)
                receipt['next_frozen_policy_inference']='GPU_REQUIRED; no old/expert/substitute action used'
            else:
                # Replay each original row once; safe subsegments retain that
                # row instead of skipping to its successor. No new query claim.
                for number,row in enumerate(rows,a.first_row):
                    row_start=backend.tick
                    while True:
                        snap=backend.capture_state()
                        result=adapter.policy_row(row,snap,remaining)
                        if not result.ok:raise G2Error(result.status,result.detail,**result.diagnostic)
                        log(dict(kind='production_plan',receipt=result.diagnostic,trajectory=asdict(result.payload)))
                        receipt['safe_segments']+=1
                        backend.execute(result.payload,log,remaining,stop_on_contact=False)
                        if backend.collision.tracker.holders:
                            receipt['stable_grasp']=True
                            break
                        if not result.diagnostic.get('partial_policy_row'):break
                        if (backend.tick-row_start)*backend.dt>=adapter.motion.max_ticks*adapter.motion.period:
                            raise RuntimeError('original row motion duration exhausted')
                    receipt['completed_rows'].append(number)
                    if backend.collision.tracker.holders:
                        receipt['stable_grasp']=True;break
        except Exception as error:
            receipt['error']=dict(detail=str(error),type=type(error).__name__,
                diagnostic=getattr(error,'diagnostic',{}),row=number)
        finally:
            tick=backend.tick;backend.hold()
            receipt.update(end_tick=tick,physical_steps=tick-receipt['start_tick'],
                hold_no_advance=backend.tick==tick,contact_state=backend.collision.tracker.snapshot(),
                holder_transitions=backend.collision.tracker.events,elapsed_s=monotonic()-started)
            save(a.output/'receipt.json',receipt)
    print('PRODUCTION_CPU',spec['job'],a.mode,receipt['physical_steps'],
          len(receipt['completed_rows']),receipt.get('error',{}).get('detail'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','audit','assets','output'):p.add_argument(name,type=Path)
    p.add_argument('--first-row',type=int,required=True)
    p.add_argument('--mode',choices=('interface','archived_tail','negative'),required=True)
    run(p.parse_args())
