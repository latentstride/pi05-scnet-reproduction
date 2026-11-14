"""Counterfactual: nominal full-row proposal, original live ARX gate at every tick.

Only the offline future stationary-object prefilter is omitted. The untouched
ARXBackend rechecks every proposed physics increment against the CURRENT object,
then checks actual collision, force contact, tracking and slip after stepping.
This is not a GPU entry point or an approved production change.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace as NS

import numpy as np

from g2_adapter import G2Error,deadline,read_snapshot,solve_pair
from g2_arx_direct import DirectAdapter
from g2_arx_transfer_probe_cpu import restart_backend
from g2_arx_endpoint_cpu import build_scene,save
from g2_motion import Trajectory
from robotwin_control_geometry import Status


class OfflineNominalProposal:
    """Unexecuted kinematic proposal; all real execution uses the original world."""
    def __init__(self,world):
        self.profile_id=world.profile_id
        self.measured_clear=world.measured_clear
        self.drive_start=world.drive_start
    def __call__(self,q,fingers):return True
    def planning_geometry(self,q,requested,previous,check):
        check()
        return requested.copy()


def feedback_row(adapter,backend,target,fingers,remaining,log):
    """Diagnostic 1-D safe steering along the SAME nominal joint segment.

    Every executed subsegment is fully replanned with the original static
    checker, original quintic limits and the unchanged live backend. No unsafe
    step is submitted. The original per-row motion tick budget bounds progress.
    """
    start_tick=backend.tick
    world=adapter.target_collision
    while backend.tick-start_tick<adapter.motion.max_ticks:
        snap=backend.capture_state();q,g,_,_=read_snapshot(snap,adapter.arx)
        try:
            plan=adapter.motion.plan(q,g,[(target,fingers)],world,remaining)
            full=True
        except G2Error as error:
            if error.status!=Status.COLLISION:raise
            nominal=adapter.motion.plan(q,g,[(target,fingers)],OfflineNominalProposal(world),remaining)
            geometry=g.copy();last=None
            for index in range(1,len(nominal.q)):
                try:
                    geometry=world.planning_geometry(nominal.q[index],nominal.fingers[index],geometry,remaining)
                    if not world(nominal.q[index],geometry):break
                except G2Error as blocked:
                    if blocked.status!=Status.COLLISION:raise
                    break
                last=index
            if last is None:
                raise G2Error(Status.COLLISION,'no safe progress on the unchanged policy joint segment',
                              original=str(error),collisions=world.last_collisions)
            plan=adapter.motion.plan(q,g,[(nominal.q[last],nominal.fingers[last])],world,remaining)
            full=False
        if backend.tick-start_tick+len(plan.q)-1>adapter.motion.max_ticks:
            raise G2Error(Status.PLAN_FAILED,'original per-row motion tick budget exhausted')
        log(dict(kind='feedback_safe_segment',full_original_target=full,
                 target=target,segment_target=plan.q[-1],trajectory=asdict(plan)))
        holders=set(world.tracker.holders)
        backend.execute(plan,log,remaining,stop_on_contact=False)
        if set(world.tracker.holders)!=holders or full:return
    raise G2Error(Status.PLAN_FAILED,'original per-row motion tick budget exhausted')


def run(a):
    a.output.mkdir()
    spec=json.loads(a.input.read_text());cfg=spec['config']
    print('DYNAMIC_ENTRY',spec['job'],flush=True)
    scene,old,snap,drive=build_scene(spec,a.audit,a.assets,a.output)
    backend,_,_=restart_backend(scene,old,snap,drive,cfg,spec['initial'],a.assets)
    actors={x.name:x for x in scene.get_all_actors()}
    backend.env.table,backend.env.target_box=actors['table'],actors['target_box']
    backend.env.get_obs=lambda:{'observation':{k:{'rgb':np.zeros((8,8,3),dtype=np.uint8)}
        for k in ('head_camera','left_camera','right_camera')}}
    backend.env.check_success=lambda:False
    backend.video=NS(frame=lambda _:None)
    adapter=DirectAdapter(old.aloha,old.arx,old.source_frames,np.eye(4),old.seed,
        old.source_speed,target_collision=old.target_collision,motion=old.motion.motion,
        grip_rule=old.grip_rule,grip_max_clip=old.grip_max_clip)
    world=adapter.target_collision
    proposed=OfflineNominalProposal(world)
    started=monotonic();check=deadline(720)
    remaining=lambda:check() or 720-(monotonic()-started)
    receipt=dict(job=spec['job'],query=spec['query'],start_tick=backend.tick,
        source_action_sha256=spec['action_sha256'],first_row=a.first_row,
        scope='CPU original policy rows; future static prefilter counterfactual only; no inference/expert/TCP offsets',
        physical_guard='original ARXBackend, original ContactWorld, original 2mm actual and per-tick candidate gates',
        completed_rows=[],holders=[],max_contact_fingers=[0,0],contact_steps=[0,0],
        stable_grasp=False,hold_0_8s=False)
    row_number=0
    with (a.output/'events.jsonl').open('x') as stream:
        def log(event):
            stream.write(json.dumps(dict(policy_row=row_number,**event),
                                    default=lambda v:v.tolist())+'\n')
            contact=event.get('contact_state') or {}
            for i,side in enumerate(('left','right')):
                n=len(contact.get('contact_fingers',{}).get(side,[]))
                receipt['max_contact_fingers'][i]=max(receipt['max_contact_fingers'][i],n)
                if n:receipt['contact_steps'][i]+=event.get('physics_step_count',0)
        try:
            snap,_=backend.capture()
            for row_number in range(a.first_row,51):
                remaining()
                if row_number>a.first_row:snap=backend.capture_state()
                q,g,frames,_=read_snapshot(snap,adapter.arx)
                values=np.array(spec['chunk'][row_number-1]).reshape(2,7)
                goals=np.array([frame@arm.chain.fk(v[:6])@arm.offset()
                    for arm,frame,v in zip(adapter.aloha,adapter.source_frames,values)])
                fingers=np.array([arm.grip.target(float(np.clip(v[6],0,1)))
                    for arm,v in zip(adapter.arx,values)])
                goals,phase=world.prepare_tools(goals,fingers,world.tools(q))
                target,ik=solve_pair(adapter.arx,frames,goals,q,fingers,
                                     lambda *args:True,deadline(45))
                if a.feedback:
                    feedback_row(adapter,backend,target,fingers,remaining,log)
                else:
                    plan=adapter.motion.plan(q,g,[(target,fingers)],proposed,check)
                    log(dict(kind='offline_nominal_proposal',ik=ik,original_policy_row=values.ravel(),
                             target_q=target,target_fingers=fingers,trajectory=asdict(plan)))
                    done=backend.execute(plan,log,remaining,stop_on_contact=False)
                receipt['completed_rows'].append(row_number)
                receipt['holders']=sorted(world.tracker.holders)
                print('DYNAMIC_ROW',row_number,backend.tick,receipt['holders'],flush=True)
                if receipt['holders']:
                    receipt['stable_grasp']=True
                    # A stationary hold is a controller verification, not a new
                    # task action or a substitute policy continuation.
                    snap=backend.capture_state();q,g,_,_=read_snapshot(snap,adapter.arx)
                    drives=backend._drive_fingers()
                    hold=Trajectory(.004,np.repeat(q[None],201,axis=0),np.zeros((201,2,6)),
                        np.repeat(drives[None],201,axis=0),(200,),
                        np.repeat(g[None],201,axis=0),world.profile_id)
                    hold_start=backend.tick
                    backend.execute(hold,log,remaining,stop_on_contact=False)
                    receipt['hold_0_8s']=(backend.tick-hold_start==200 and
                        sorted(world.tracker.holders)==receipt['holders'])
                    break
        except Exception as error:
            receipt['error']=dict(type=type(error).__name__,detail=str(error),
                status=getattr(getattr(error,'status',None),'value',None),
                diagnostic=getattr(error,'diagnostic',{}),row=row_number)
        finally:
            tick=backend.tick;backend.hold()
            receipt.update(end_tick=tick,physics_steps=tick-receipt['start_tick'],
                hold_no_advance=backend.tick==tick,contact_state=world.tracker.snapshot(),
                holder_transitions=world.tracker.events,elapsed_s=monotonic()-started)
            save(a.output/'receipt.json',receipt)
    print('DYNAMIC_COMPLETE',spec['job'],receipt['stable_grasp'],receipt['hold_0_8s'],
          receipt.get('error',{}).get('detail'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','audit','assets','output'):p.add_argument(name,type=Path)
    p.add_argument('--first-row',required=True,type=int)
    p.add_argument('--feedback',action='store_true')
    run(p.parse_args())
