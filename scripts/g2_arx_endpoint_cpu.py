"""Fixed-state source/target mesh diagnosis. No inference, execution or runtime gate changes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np


def select_rows(chunk, rows):
    chunk=np.asarray(chunk,dtype=float)
    if chunk.shape!=(50,14) or not np.isfinite(chunk).all():
        raise ValueError('expected a finite original 50x14 policy block')
    if not rows or len(set(rows))!=len(rows) or any(type(n) is not int or not 1<=n<=50 for n in rows):
        raise ValueError('row numbers must be distinct integers in 1..50')
    return [(n,chunk[n-1].copy()) for n in rows]


def depth_mm(depth):
    return None if depth is None else float(depth)*1000


def build_scene(spec,audit,assets,output):
    from arx_replay import build_adapter
    return build_adapter(spec['config'],assets,audit,spec['initial'],spec['at'],
                         spec['request'],output)


def save(path,receipt):
    path.write_text(json.dumps(receipt,indent=2,allow_nan=False,
        default=lambda value:value.tolist())+'\n')


def diagnose(spec,audit,assets,output,rows):
    output.mkdir(parents=True,exist_ok=False)
    receipt=dict(job=spec.get('job'),query=spec.get('query'),status='running',
        scope='fixed original policy endpoints; no physics advancement or inference',
        source_action_sha256=spec.get('action_sha256'),records=[])
    phase='validate_input'
    try:
        selected=select_rows(spec['chunk'],rows)
        phase='build_scene'
        save(output/'receipt.json',dict(receipt,phase=phase))
        print('ENDPOINT_PHASE',phase,flush=True)
        scene,adapter,snap,drive=build_scene(spec,audit,assets,output)
        from g2_adapter import G2Error,deadline,read_snapshot,solve_pair
        q,g,frames,_=read_snapshot(snap,adapter.arx)
        receipt['snapshot_stamp']=vars(snap.stamp)
        receipt['asset_audit_sha256']=sha256(audit.read_bytes()).hexdigest()
        source_g=np.array([arm.grip.target(target.grip.read(v))
                           for arm,target,v in zip(adapter.aloha,adapter.arx,g)])
        for number,row_flat in selected:
            phase=f'endpoint_row_{number}'
            print('ENDPOINT_PHASE',phase,flush=True)
            row=row_flat.reshape(2,7)
            source_requested=np.array([arm.grip.target(float(np.clip(v[6],0,1)))
                                       for arm,v in zip(adapter.aloha,row)])
            requested=np.array([arm.grip.target(float(np.clip(v[6],0,1)))
                                for arm,v in zip(adapter.arx,row)])
            entry=dict(row=number,original_action=row_flat.tolist(),gripper_intent=row[:,6].tolist())
            entry['source_start_finger_depth_mm']=[[
                depth_mm(adapter.source_collision.finger_depth(row[:,:6],source_g,s,f))
                for f in range(2)] for s in range(2)]
            entry['source_requested_finger_depth_mm']=[[
                depth_mm(adapter.source_collision.finger_depth(row[:,:6],source_requested,s,f))
                for f in range(2)] for s in range(2)]
            goals=np.array([frame@arm.chain.fk(v[:6])@arm.offset()
                            for arm,frame,v in zip(adapter.aloha,adapter.source_frames,row)])
            entry['original_tools_task']=goals.tolist()
            try:
                # Diagnostic construction only. Every actual production verdict
                # is separately evaluated below; nothing reaches an executor.
                target,residual=solve_pair(adapter.arx,frames,goals,q,requested,
                                           lambda *args:True,deadline(45))
                entry.update(kinematic_ik=residual,target_q=target.tolist())
                entry['target_start_finger_depth_mm']=[[
                    depth_mm(adapter.target_collision.finger_depth(target,g,s,f))
                    for f in range(2)] for s in range(2)]
                entry['target_requested_finger_depth_mm']=[[
                    depth_mm(adapter.target_collision.finger_depth(target,requested,s,f))
                    for f in range(2)] for s in range(2)]
                entry['raw_requested_endpoint_clear']=bool(adapter.target_collision(target,requested))
                entry['raw_endpoint_collisions']=adapter.target_collision.last_collisions
                adapter.target_collision.set_start(g,drive)
                entry['production_ik_clear']=bool(adapter.target_collision.ik_clear(target,requested,deadline(45)))
                entry['production_failure']=getattr(adapter.target_collision,'failure',None)
            except G2Error as error:
                entry['error']=dict(status=error.status.value,detail=str(error),diagnostic=error.diagnostic)
            receipt['records'].append(entry)
            save(output/'receipt.json',dict(receipt,phase=phase))
        receipt.update(status='complete',phase='complete',physics_steps=0)
        save(output/'receipt.json',receipt)
        print('ENDPOINT_COMPLETE',spec.get('job'),rows,flush=True)
        return 0
    except Exception as error:
        receipt.update(status='error',failed_phase=phase,error=repr(error),traceback=traceback.format_exc())
        save(output/'receipt.json',receipt)
        print('ENDPOINT_ERROR',phase,repr(error),flush=True)
        return 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','audit','assets','output'):p.add_argument(name,type=Path)
    p.add_argument('--rows',type=int,nargs='+',required=True)
    a=p.parse_args()
    print('ENDPOINT_ENTRY',flush=True)
    spec=json.loads(a.input.read_text())
    raise SystemExit(diagnose(spec,a.audit,a.assets,a.output,a.rows))


if __name__=='__main__':main()
