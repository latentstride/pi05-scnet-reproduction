"""Real-mesh planning gate acceptance; CPU physics, synthetic cameras, no policy."""
import argparse,json,time,signal,copy
from contextlib import contextmanager,nullcontext
from dataclasses import asdict
from pathlib import Path
import numpy as np
from g2_adapter import Stamp,Snapshot,deadline,read_snapshot,G2Error
from g2_capture import capture_arms
from g2_episode import save_json,ExpertWaypoints
from g2_gate_scene_cpu import build
from g2_robotwin_backend import pose7
from g2_run_task import execute
from g2_arx_expert_monitor_cpu_v5 import unpack_trace
from robotwin_control_geometry import inverse_pose,pose_from_wxyz

def finger_point(scene):
    link=scene.entities[0].find_joint_by_name(scene.arms[0].grip.names[0]).child_link
    shape=link.collision_shapes[0]
    point=np.mean(np.asarray(shape.vertices)*np.asarray(shape.scale),axis=0)
    return (link.entity.pose.to_transformation_matrix()@shape.local_pose.to_transformation_matrix()@np.r_[point,1])[:3]

def run(args):
    import sapien
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();checks={};records=[]
    def emit(name,passed,**detail):
        checks[name]=bool(passed);record=dict(name=name,passed=bool(passed),**detail);records.append(record)
        save_json(args.output/'gate-cases.json',records);print('GATE_CASE',name,bool(passed),flush=True)
    def make(name):return build(args.assets,args.audit,args.config,args.input,args.reference,args.output/name)
    measured=np.load(args.measured/'physics-trace.npz',allow_pickle=False)
    stages=json.loads((args.measured/'stages.json').read_text())
    s=make('geometry');robot=s.robot;tracker=s.tracker
    initial_q=[e.get_qpos().copy() for e in s.entities];initial_v=[e.get_qvel().copy() for e in s.entities]
    initial_bases=[e.get_root_pose() for e in s.entities];initial_block=s.block.get_pose();initial_target=s.target.get_pose()
    original_tracker=copy.deepcopy(tracker.__dict__)
    def reset():
        for e,q,v,p in zip(s.entities,initial_q,initial_v,initial_bases):e.set_root_pose(p);e.set_qpos(q);e.set_qvel(v)
        s.block.set_pose(initial_block);s.target.set_pose(initial_target);s.probe.set_pose(sapien.Pose([3,3,3]))
        tracker.__dict__.clear();tracker.__dict__.update(copy.deepcopy(original_tracker));s.contact.refresh()
    def state():return read_snapshot(Snapshot(Stamp('geometry',0,0.),capture_arms(robot,Stamp('geometry',0,0.)),{},np.eye(4)),s.arms)
    def load_checkpoint(tick,holders):
        reset();i=int(np.searchsorted(measured['sample_ticks'],tick));native=measured['actual_native'][i]
        for side,(e,arm) in enumerate(zip(s.entities,s.arms)):
            names=[j.name for j in e.get_active_joints()];q=e.get_qpos().copy()
            for name,value in zip(arm.chain.names+arm.grip.names,native[side*8:(side+1)*8]):q[names.index(name)]=value
            e.set_qpos(q);e.set_qvel(np.zeros(e.dof))
        value=measured['block_pose'][i];s.block.set_pose(sapien.Pose(value[:3],value[3:]));s.contact.refresh()
        q,g,frames,tools=state();bm=s.block.pose.to_transformation_matrix()
        tracker.holders={side:inverse_pose(tools[('left','right').index(side)])@bm for side in holders}
        tracker.measured_attachments={k:v.copy() for k,v in tracker.holders.items()};tracker.block_pose=bm.copy()
        tracker.ever_held=True;tracker.saw_shared_hold=len(holders)==2
        drive=measured['drive_native'][i].reshape(2,8)[:,6:]
        s.contact.set_start(g,drive)
        return q,g,tools,drive,i
    def compare_projection(q,drive,label):
        original=s.contact.object_candidate;original_context=s.contact.projection_context;entries=[]
        @contextmanager
        def counted(q):
            entries.append(True)
            with original(q):yield
        s.contact.object_candidate=counted
        try:
            started=time.monotonic();optimized,changes=s.contact.project_fingers(q,drive,deadline(60))
            optimized_s=time.monotonic()-started;optimized_entries=len(entries);entries.clear()
            s.contact.projection_context=lambda q:nullcontext()
            started=time.monotonic();legacy,old_changes=s.contact.project_fingers(q,drive,deadline(60))
            legacy_s=time.monotonic()-started;legacy_entries=len(entries)
            emit('projection_equivalent_'+label,np.array_equal(optimized,legacy) and changes==old_changes and optimized_entries==1 and legacy_entries>1,
                optimized_fingers=optimized,legacy_fingers=legacy,optimized_object_contexts=optimized_entries,legacy_object_contexts=legacy_entries,
                optimized_s=optimized_s,legacy_s=legacy_s)
            return optimized,changes
        finally:s.contact.object_candidate=original;s.contact.projection_context=original_context
    try:
        reference_initial=json.loads((args.measured/'initial.json').read_text())
        joint_error=float(np.max(abs(np.concatenate(initial_q)-np.array(reference_initial['actual_all_qpos']))))
        block_error=float(np.max(abs(np.r_[s.block.pose.p,s.block.pose.q]-np.array(reference_initial['block_pose']))))
        emit('original_10g_initial_matches_record',joint_error<1e-6 and block_error<1e-6,joint_max_error=joint_error,block_pose_max_error=block_error,actual_mass_kg=s.initial['actual_block_mass_kg'])
        q,g,_,_=state();s.contact.set_start(g,g)
        emit('initial_measured_geometry_clear',s.contact.measured_clear(q,g),blocked=s.contact.last_collisions)
        # Explicit cross-articulation collision, including coincident fixed bases.
        s.entities[1].set_root_pose(initial_bases[0]);s.contact.refresh()
        clear=s.contact.measured_clear(q,g);rows=copy.deepcopy(s.checker.last_collisions)
        cross=any(r['object_name1']!=r['object_name2'] and r['object_name1'] in s.contact.model_names and r['object_name2'] in s.contact.model_names for r in rows)
        emit('cross_arm_collision_rejected',not clear and cross,blocked=rows);reset()
        q,g,_,_=state();s.contact.set_start(g,g);s.target.set_pose(sapien.Pose(finger_point(s)));s.contact.refresh()
        clear=s.contact.measured_clear(q,g);rows=copy.deepcopy(s.checker.last_collisions)
        emit('robot_tray_collision_rejected',not clear and any('target_box' in (r['object_name1']+r['object_name2']) for r in rows),blocked=rows);reset()
        # Historical measured checkpoints are kinematic geometry fixtures, not new dynamic grasps.
        q,g,tools,drive,i=load_checkpoint(stages[3]['tick'],['left'])
        raw=s.checker(q,drive);raw_rows=copy.deepcopy(s.checker.last_collisions)
        predicted,changes=compare_projection(q,drive,'single_holder');projected=s.contact(q,predicted)
        emit('closing_intent_separated_from_geometry',not raw and projected and np.any(abs(predicted-drive)>.001),
            checkpoint_tick=int(measured['sample_ticks'][i]),raw_drive=drive,predicted_fingers=predicted,changes=changes,raw_blocked=raw_rows,projected_blocked=s.contact.last_collisions)
        object_pose=s.checker.world.get_object(s.contact.block_name).pose
        before=pose_from_wxyz(object_pose.p,object_pose.q)
        candidate=q.copy();candidate[0,0]+=.12;position=s.contact.held_pose_world(candidate)[:3,3]
        s.probe.set_pose(sapien.Pose(position));s.contact.refresh();verdict=s.contact(candidate,predicted)
        rows=copy.deepcopy(s.checker.last_collisions)
        emit('carried_object_obstacle_rejected',not verdict and any(r['collision_type']=='held_object_environment' and 'gate_probe' in (r['object_name1']+r['object_name2']) for r in rows),blocked=rows)
        object_pose=s.checker.world.get_object(s.contact.block_name).pose
        after=pose_from_wxyz(object_pose.p,object_pose.q)
        emit('candidate_object_pose_restored',np.allclose(before,after,atol=1e-9,rtol=0))
        s.probe.set_pose(sapien.Pose([3,3,3]));s.contact.refresh()
        opening=drive.copy();opening[0]=.044;early=False
        try:tracker.guard_opening(drive,opening,s.arms)
        except G2Error as error:early='before receiver' in str(error)
        emit('premature_donor_release_rejected',early)
        q,g,tools,drive,i=load_checkpoint(stages[8]['tick'],['left','right'])
        compare_projection(q,drive,'two_holders')
        requested=drive.copy();requested[0]=.044;goals=np.array(tools);goals[0,0,3]+=.05
        prepared,phase=s.contact.prepare_tools(goals,requested,tools)
        emit('release_before_withdrawal_barrier',phase=='donor_release_before_withdrawal' and np.array_equal(prepared,tools))
        candidate=q.copy();candidate[1,0]+=.08;verdict=s.contact(candidate,g)
        emit('inconsistent_two_holder_chain_rejected',not verdict and 'inconsistent object' in str(s.contact.last_collisions),blocked=s.contact.last_collisions)
        reset()
    except BaseException as error:
        emit('geometry_suite_completed',False,error_type=type(error).__name__,detail=str(error))
        raise
    finally:
        reset()
    packed=unpack_trace(args.trace);expert_rows=packed['target'][packed['frame_ticks']]
    config=json.loads(args.config.read_text())
    # A valid computed plan becomes unsafe before execute: test the actual episode cleanup path.
    negative={};negative_hold=[]
    @contextmanager
    def negative_factory(spec):
        test=make('changed-scene');negative['scene']=test;original=test.backend.hold
        def hold():negative_hold.append(True);return original()
        test.backend.hold=hold
        class ChangedScene:
            def begin_expert(self,*a):return test.adapter.begin_expert(*a)
            def actions(self,*a):
                result=test.adapter.actions(*a)
                if result.ok:
                    test.probe.set_pose(sapien.Pose(finger_point(test)))
                    negative['planned_before_injection']=True
                return result
        try:yield test.backend,ChangedScene()
        finally:negative['factory_closed']=True
    cfg=dict(config,episode_id='gate-scene-change',max_actions=1,timeout_s=90.)
    result=execute(cfg,negative_factory,None,args.output/'changed-scene-episode',ExpertWaypoints(expert_rows,'fixed expert source; scene mutation test'))
    test=negative['scene'];events=[json.loads(line) for line in (args.output/'changed-scene-episode/events.jsonl').read_text().splitlines()]
    rejections=[r for r in events if r.get('kind')=='planning_gate_rejection']
    stopped=all(np.allclose(j.get_drive_velocity_target(),0.) for e in test.entities for j in e.get_active_joints())
    emit('changed_scene_rejected_before_actuation',negative.get('planned_before_injection') and result['outcome']=='control_error' and test.backend.tick==0 and result['control_calls']==0,
        result=result,physics_ticks=test.backend.tick)
    emit('rejection_collision_rows_saved',bool(rejections and rejections[0]['collisions']),rejection_events=rejections)
    emit('rejection_dual_hold_and_cleanup',negative_hold==[True] and stopped and bool(test.closed) and negative.get('factory_closed'),hold_calls=len(negative_hold),zero_velocity_targets=stopped)
    # Run the unchanged production expert path until success, a concrete rejection, or a declared deadline.
    live={}
    @contextmanager
    def factory(spec):
        test=make('live');live['scene']=test
        try:yield test.backend,test.adapter
        finally:live['factory_closed']=True
    cfg=dict(config,episode_id='gate-live-expert',max_actions=len(expert_rows),timeout_s=args.live_timeout)
    result=execute(cfg,factory,None,args.output/'live-episode',ExpertWaypoints(expert_rows,'fixed official Aloha recorded-target sequence'))
    live_scene=live['scene']
    emit('valid_production_prefix_advanced',result['executed']>=3 and live_scene.backend.tick>0,executed_actions=result['executed'],physics_ticks=live_scene.backend.tick,outcome=result['outcome'])
    report=dict(data_kind='diagnostic',gpu_used=False,renderer_created=False,model_used=False,synthetic_cameras=True,
        functional_gate_passed=all(checks.values()),expert_full_accepted=result['success'],passed=all(checks.values()) and result['success'],
        checks=checks,cases=records,live_result=result,live_physics_steps=live_scene.backend.tick,contact_state=live_scene.tracker.snapshot(),
        scope='real mesh FCL and production adapter/motion/backend/episode on CPU; no rendering or policy evaluation',
        geometry_fixture_kind='archived actual joint/object states restored kinematically; holder assumptions from recorded handover phases',
        contact_limits=asdict(live_scene.tracker.limits),physical_filter=s.physical_profile,wall_s=time.monotonic()-start)
    save_json(args.output/'acceptance.json',report);print('GATE_FINAL',json.dumps({k:report[k] for k in ['functional_gate_passed','expert_full_accepted','passed','live_physics_steps','wall_s']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','audit','config','input','reference','trace','measured','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--live-timeout',type=float,default=300.)
    a=p.parse_args()
    def terminate(signum,frame):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,terminate)
    run(a)
