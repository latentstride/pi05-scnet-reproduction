"""Bounded old/new mapping comparison at archived requests 35 and 36, CPU only."""
import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
from arx_replay import build_adapter, digest, load, summary_error, EPISODE
from g2_adapter import G2Error, deadline, read_snapshot, solve_pair, Snapshot, Stamp
from g2_capture import capture_arms
from g2_collision import apply_audited_arx_physics
from g2_remaining_coordinator import RecoveryCoordinator
from g2_robotwin_backend import ARXBackend
from g2_task_tcp_projection import plan_projection


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=lambda x:x.tolist())+'\n')


def restart_backend(scene, adapter, snap, drive, config, initial, assets):
    """Use the existing guarded backend, declaring unavailable historical velocities."""
    import sapien
    from g2_static_failure_replay import matrix
    entities={e.name:e for e in scene.get_all_articulations()}
    pair=[entities['arx_left'],entities['arx_right']]
    physical=apply_audited_arx_physics(scene,pair,assets)
    for entity in pair:
        for link in entity.get_links():link.set_mass(1.)
        for joint,value in zip(entity.get_active_joints(),entity.get_qpos()):
            joint.set_drive_property(stiffness=1000.,damping=200.,mode='force')
            joint.set_drive_target(float(value));joint.set_drive_velocity_target(0.)
        entity.set_qvel(np.zeros(entity.dof))
    robot=NS(left_entity=pair[0],right_entity=pair[1])
    block=next(x for x in scene.get_all_actors() if x.name=='box')
    env=NS(robot=robot,scene=scene,box=block,take_action_cnt=0)
    backend=ARXBackend(env,adapter.arx,adapter.target_collision,matrix(config['world_task']),.004,
                       initial,config['tracking_position_limit_m'],config['tracking_angle_limit_deg'],
                       NS(),contact_drive=config['contact_profile']['drive'])
    backend.episode=snap.stamp.episode;backend.tick=snap.stamp.tick;backend.last_capture=snap
    for (_,_,grip),values in zip(backend._joints(),drive):
        for joint,value in zip(grip,values):joint.set_drive_target(float(value))
    return backend, robot, physical


def dynamic_restart(scene, adapter, snap, drive, plan, config, initial, assets, output):
    backend, robot, physical = restart_backend(scene, adapter, snap, drive, config, initial, assets)
    peak=[0.,0.];counts={}
    with (output/'dynamic-events.jsonl').open('w') as stream:
        def log(row):
            stream.write(json.dumps(row,default=lambda x:x.tolist())+'\n')
            if 'position_error_m' in row:peak[0]=max(peak[0],max(row['position_error_m']))
            if 'orientation_error_deg' in row:peak[1]=max(peak[1],max(row['orientation_error_deg']))
            phase=(row.get('contact_state') or {}).get('phase')
            if phase:counts[phase]=counts.get(phase,0)+1
        try:
            done=backend.execute(plan,log,deadline(120.),stop_on_success=False)
            result=dict(accepted=done==len(plan.waypoint_ticks),completed_actions=done)
        except G2Error as error:
            result=dict(accepted=False,error=summary_error(error))
    stamp=Stamp(snap.stamp.episode,backend.tick,backend.tick*.004)
    fresh=Snapshot(stamp,capture_arms(robot,stamp),{},snap.world_task)
    q,g,frames,tools=read_snapshot(fresh,adapter.arx)
    world=adapter.source_collision;world.refresh()
    fractions=[a.grip.measure(v)['normalized'] for a,v in zip(adapter.arx,g)]
    source_fingers=np.array([a.grip.target(v) for a,v in zip(adapter.aloha,fractions)])
    world.set_start(source_fingers)
    try:
        proxy,residual=solve_pair(adapter.aloha,adapter.source_frames,tools,adapter.seed,
                                  source_fingers,world,deadline(30),adapter.source_speed*((backend.tick-snap.stamp.tick)*.004))
        result['real_state_source_reobservation']=dict(accepted=True,q=proxy,residual=residual)
    except G2Error as error:result['real_state_source_reobservation']=dict(accepted=False,error=summary_error(error))
    backend.hold()
    result.update(physics_steps=backend.tick-snap.stamp.tick,peak_tracking_position_m=peak[0],
                  peak_tracking_angle_deg=peak[1],phases=counts,physical_filter=physical,
                  restart_scope='archived measured q/fingers/object pose; zero velocities; no historical velocity or solver-contact cache restoration',
                  final_q=q,final_fingers=g,final_tools=tools,hold_called=True)
    return result


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    folder=a.run/'trials'/EPISODE/'episode'; publication=load(a.publication)
    for name in ('events.jsonl','action-00035.npy','action-00036.npy','result.json'):
        if digest(folder/name)!=publication['source_sha256'][f'trials/{EPISODE}/episode/{name}']:
            raise ValueError('archived input hash differs: '+name)
    config=load(a.release/'configs'/f'{EPISODE}.json');initial=load(folder/'result.json')['initial']
    requests=[];states={};step=None
    for line in (folder/'events.jsonl').open():
        row=json.loads(line)
        if row['kind']=='executed_step':step=row
        if row['kind']=='policy_request':
            requests.append(row)
            if len(requests) in (35,36):
                assert step['tick']==row['stamp']['tick'];states[len(requests)]=step
    result=dict(gpu_jobs_submitted=0,model_inference=False,expert_actions_used=False,
                candidate_limit=18,independent_historical_snapshots=True,reports=[])
    for query in (35,36):
        out=a.output/str(query);out.mkdir()
        scene,adapter,snap,drive=build_adapter(config,a.assets,a.release/'asset-audit.json',initial,states[query],requests[query-1],out)
        chunk=np.load(folder/f'action-{query:05d}.npy',allow_pickle=False)
        report=dict(query=query,tick=snap.stamp.tick,input_action_sha256=digest(folder/f'action-{query:05d}.npy'))
        raw=[]
        for i,row in enumerate(chunk[:4].reshape(4,2,7)):
            fingers=np.array([a.grip.target(float(np.clip(v[6],0,1))) for a,v in zip(adapter.aloha,row)])
            adapter.source_collision.set_start(fingers)
            clear=adapter.source_collision.ik_clear(row[:,:6],fingers,deadline(20))
            raw.append(dict(row=i+1,accepted=bool(clear),collisions=adapter.source_collision.last_collisions))
        report['raw_policy_source_geometry']=raw
        old=RecoveryCoordinator(adapter,NS(_drive_fingers=lambda:drive),max_recoveries=4,approach_repair=True)
        try:
            plan,receipt=old.action_plan(chunk,snap,4,lambda:90.,lambda row:None)
            report['old']=dict(approved_actions=len(plan.waypoint_ticks),receipt=receipt)
        except G2Error as error:report['old']=dict(approved_actions=0,error=summary_error(error))
        # Rebuild so old planning cannot leak mutable collision prediction state.
        del old,adapter,scene
        fresh=out/'new';fresh.mkdir()
        scene,adapter,snap,drive=build_adapter(config,a.assets,a.release/'asset-audit.json',initial,states[query],requests[query-1],fresh)
        try:
            plan,receipt=plan_projection(adapter,snap,chunk,4,180.)
            report['new']=dict(approved_actions=len(plan.waypoint_ticks),receipt=receipt)
            np.savez_compressed(fresh/'plan.npz',**{k:v for k,v in asdict(plan).items() if v is not None})
        except G2Error as error:
            report['new']=dict(approved_actions=0,error=summary_error(error),diagnostic=error.diagnostic)
        else:
            report['dynamic']=dynamic_restart(scene,adapter,snap,drive,plan,config,initial,a.assets,fresh)
        result['reports'].append(report);save(a.output/'comparison.json',result)
        print('CPU_TRANSFER',query,'old',report['old']['approved_actions'],'new',report['new']['approved_actions'],flush=True)
        del adapter,scene
    result['decision']='NOT_READY'
    result['decision_scope']='CPU candidate evidence only; no live policy after intervention and no source/target initial-state matched transfer acceptance'
    save(a.output/'comparison.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','release','assets','publication','output'):p.add_argument(name,type=Path)
    run(p.parse_args())
