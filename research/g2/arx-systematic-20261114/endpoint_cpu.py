"""Diagnose contact semantics at fixed recorded targets; never execute an unsafe pose."""

import argparse
import json
from pathlib import Path

import numpy as np

from arx_replay import build_adapter, load
from g2_adapter import G2Error, deadline, read_snapshot, solve_pair


def run(a):
    a.output.mkdir()
    spec=load(a.input);cfg=spec['config']
    scene, old, snap, drive=build_adapter(cfg,a.assets,a.audit,spec['initial'],
        spec['at'],spec['request'],a.output)
    q,g,frames,_=read_snapshot(snap,old.arx)
    chunk=np.asarray(spec['chunk'])
    records=[]
    for number in (1,8,26,27,50):
        row=chunk[number-1].reshape(2,7)
        goals=np.array([f@arm.chain.fk(v[:6])@arm.offset()
                        for arm,f,v in zip(old.aloha,old.source_frames,row)])
        requested=np.array([arm.grip.target(float(np.clip(v[6],0,1)))
                            for arm,v in zip(old.arx,row)])
        entry=dict(row=number,gripper_intent=row[:,6].tolist())
        try:
            if a.source_only:
                source_g=np.array([arm.grip.target(target.grip.measure(v)['normalized'])
                                   for arm,target,v in zip(old.aloha,old.arx,g)])
                source_requested=np.array([arm.grip.target(float(np.clip(v[6],0,1)))
                                           for arm,v in zip(old.aloha,row)])
                entry['source_start_finger_depth_mm']=[[
                    old.source_collision.finger_depth(row[:,:6],source_g,s,f)*1000
                    for f in range(2)] for s in range(2)]
                entry['source_requested_finger_depth_mm']=[[
                    old.source_collision.finger_depth(row[:,:6],source_requested,s,f)*1000
                    for f in range(2)] for s in range(2)]
                records.append(entry)
                continue
            # Deliberately bypass only the collision callback for diagnostic FK/IK
            # construction. This candidate is never submitted to the executor.
            target,residual=solve_pair(old.arx,frames,goals,q,requested,
                                      lambda *args: True,deadline(45))
            entry['kinematic_ik']=residual
            entry['start_finger_depth_mm']=[[old.target_collision.finger_depth(target,g,s,f)*1000
                                          for f in range(2)] for s in range(2)]
            entry['requested_finger_depth_mm']=[[old.target_collision.finger_depth(target,requested,s,f)*1000
                                          for f in range(2)] for s in range(2)]
            entry['raw_requested_endpoint_clear']=bool(old.target_collision(target,requested))
            entry['raw_requested_endpoint_collisions']=old.target_collision.last_collisions
            old.target_collision.set_start(g,drive)
            entry['production_endpoint_ik_clear']=bool(old.target_collision.ik_clear(target,requested,deadline(45)))
            entry['production_failure']=getattr(old.target_collision,'failure',None)
        except G2Error as error:
            entry['error']=dict(status=error.status.value,detail=str(error),diagnostic=error.diagnostic)
        records.append(entry)
    (a.output/'receipt.json').write_text(json.dumps(dict(job=spec['job'],query=spec['query'],
        scope='fixed endpoints only; no physical advancement; no policy inference; no runtime gate changes',
        records=records),indent=2,default=lambda v:v.tolist(),allow_nan=False)+'\n')
    print('ENDPOINT_CPU',[(v['row'],v.get('production_endpoint_ik_clear'),v.get('start_finger_depth_mm'))
                          for v in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','audit','assets','output'):p.add_argument(key,type=Path)
    p.add_argument('--source-only',action='store_true')
    run(p.parse_args())
