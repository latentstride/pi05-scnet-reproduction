"""Verify task outcome, actual state and continuous contact guard for ARX references."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from g2_arx_reference import retime_reference
from g2_kinematics import pose_error
from g2_contact import finger_force_limits
from robotwin_control_geometry import pose_from_wxyz

def first_failure_metrics(monitor):
    """Recover the instantaneous breach, independently of downsampled maxima."""
    failure=monitor.get('failure')
    if failure is None:return None
    diagnostic=failure.get('diagnostic',{});side=diagnostic.get('side');state=diagnostic.get('contact_state',{})
    if side not in state.get('tool_block',{}) or side not in state.get('tool_block_measured',{}):return None
    error=pose_error(np.array(state['tool_block'][side]),np.array(state['tool_block_measured'][side]))
    return dict(tick=failure['tick'],sim_s=failure['tick']*.004,side=side,
                translation_mm=float(np.linalg.norm(error[:3])*1000),angle_deg=float(np.rad2deg(np.linalg.norm(error[3:]))))

def verify(directory,reference,source_root,output,expected_force_n=2.,episodes=(0,1,2)):
    archive=np.load(reference,allow_pickle=False);rows=[]
    for ep in episodes:
        folder=directory/f'episode{ep}';result=json.loads((folder/'replay.json').read_text());monitor=json.loads((folder/'contact-monitor.json').read_text());trace=np.load(folder/'physics-trace.npz',allow_pickle=False)
        source=np.load(source_root/f'expert-dense-episode{ep}-20261106/trace.npz',allow_pickle=False)
        acceleration=result['retiming'].get('arm_acceleration_limit_rad_s2',1.)
        planned,stats=retime_reference(archive['q'+str(ep)],archive['g'+str(ep)],source['frame_ticks'],arm_acceleration=acceleration);n=len(planned['target'])-1
        sampled=trace['sample_ticks'];intent=planned['target'][np.minimum(sampled,n)]
        measured_arm_drive=np.c_[trace['drive_native'][:,:6],trace['drive_native'][:,8:14]]
        expected_arm=np.c_[intent[:,:6],intent[:,7:13]]
        drive_error=float(np.max(abs(measured_arm_drive-expected_arm)))
        finger_error=float(np.max(abs(trace['drive_native'][:,[6,7,14,15]]-np.repeat(.044*intent[:,[6,13]],2,axis=1))))
        actual_fingers=trace['actual_native'][:,[6,7,14,15]]
        workspace=json.loads((source_root/f'expert-workspace-20261106-r2/episode{ep}.json').read_text())
        sample_indices=np.searchsorted(sampled,trace['frame_ticks'])
        if not np.array_equal(sampled[sample_indices],trace['frame_ticks']):raise ValueError('missing measured waypoint')
        measured_tools=trace['actual_endpose_matrix'][sample_indices]
        tool_errors=np.array([[pose_error(np.array(row['target_world_tcp']),actual) for row,actual in zip(record['arms'],tools)] for record,tools in zip(workspace['records'],measured_tools)])
        initial=json.loads((folder/'initial.json').read_text());target=initial['target_pose'];top=(pose_from_wxyz(target[:3],target[3:])@np.array([0,0,.005,1]))[:3]
        predicates_consistent=True
        for physical_tick,key in [(n,'native_success_at_expert_end'),(n+200,'native_success_after_diagnostic_hold')]:
            idx=np.flatnonzero(sampled==physical_tick)
            if len(idx)!=1:predicates_consistent=False;continue
            idx=int(idx[0]);v=trace['block_pose'][idx];bottom=(pose_from_wxyz(v[:3],v[3:])@np.array([0,0,-.1,1]))[:3];delta=bottom-top
            verdict=bool(np.all(abs(delta[:2])<.03) and abs(delta[2])<.01 and trace['normalized_gripper_target'][idx,1]>.8)
            saved=result[key];predicates_consistent &= saved is not None and verdict==saved['success'] and bool(np.allclose(delta,saved['delta_m'],atol=1e-8,rtol=0))
        phases=[e['after'] for e in monitor['events']]
        ordered=False
        for i,p in enumerate(phases):
            if p==['left']:
                for j in range(i+1,len(phases)):
                    if phases[j]==['left','right'] and any(v==['right'] for v in phases[j+1:]):ordered=True
        full_contacts=trace['contacts_4ms'];positions=np.array(stats['max_command_velocity']);accelerations=np.array(stats['max_command_acceleration'])
        checks=dict(completed=result['completed'] and result['physics_steps_replayed']==n,
            source_mass=bool(abs(result['actual_block_mass_kg']-.01)<1e-8),force_cap=bool(np.array_equal(finger_force_limits(result['gripper_force_limit_n']),finger_force_limits(expected_force_n))),
            task_predicate_matches_measured_trace=bool(predicates_consistent),drive_values_match_reference=drive_error<=2e-7 and finger_error<=1e-8,
            complete_contact_record=len(full_contacts)==n+201,
            contact_monitor_continuous=monitor.get('continuous_guard_ticks')==1 and len(full_contacts)==n+201 and monitor['failure'] is None,
            contact_monitor_passed=result.get('contact_monitor_passed',False) and monitor['failure'] is None,
            handover_contact_order=ordered,finite_actual=bool(np.isfinite(trace['actual_native']).all() and np.isfinite(trace['block_pose']).all()),
            command_speed_bounds=bool(np.all(positions[:,:6]<=.5+1e-10) and np.all(positions[:,6:]<=.03+1e-10)),
            command_acceleration_bounds=bool(np.all(accelerations[:,:6]<=acceleration+1e-10) and np.all(accelerations[:,6:]<=.1+1e-10)),
            source_waypoints_retained=bool(np.array_equal(planned['frame_ticks'],trace['frame_ticks'])),
            finger_hard_limits=bool(np.min(actual_fingers)>=-.001 and np.max(actual_fingers)<=.045))
        force_max=None
        if result.get('finger_force_control_mode'):
            from g2_finger_force import MODE
            force_values=trace['finger_force_4ms'];caps=finger_force_limits(expected_force_n)
            checks['force_controller_identity']=result['finger_force_control_mode']==MODE
            checks['all_control_forces_bounded']=bool(force_values.shape==(n+201,2,2) and np.isfinite(force_values).all() and
                np.all(np.abs(force_values)<=caps[None,:,None]+1e-7))
            force_max=np.max(abs(force_values),axis=0).tolist()
        rows.append(dict(episode=ep,checks=checks,verification_passed=all(checks.values()),task_success=bool(result['completed'] and result['native_success_at_expert_end']['success']),hold_success=result['native_success_after_diagnostic_hold']['success'] if result['completed'] and result['native_success_after_diagnostic_hold'] else None,
            retimed_physics_steps=n,retimed_duration_s=stats['retimed_duration_s'],wall_s=result['wall_s'],functional_point_delta_mm=(np.array(result['native_success_at_expert_end']['delta_m'])*1000).tolist(),
            sampled_max_cumulative_slip_mm=monitor['max_cumulative_slip_m']*1000,sampled_max_cumulative_slip_deg=monitor['max_cumulative_slip_deg'],slip_statistics_period_ticks=monitor.get('max_slip_statistics_sample_ticks'),
            max_actual_finger_target_penetration_mm=float(full_contacts[:,7].max()*1000),max_actual_tool_position_error_mm=float(np.linalg.norm(tool_errors[:,:,:3],axis=2).max()*1000),max_actual_tool_angle_error_deg=float(np.rad2deg(np.linalg.norm(tool_errors[:,:,3:],axis=2)).max()),drive_max_error_rad=drive_error,drive_max_finger_error_m=finger_error,
            gripper_force_limit_n=result['gripper_force_limit_n'],monitor_failure=monitor['failure'],first_failure_metrics=first_failure_metrics(monitor),guard_observed_through_tick=monitor['failure']['tick'] if monitor['failure'] else n+200,contact_events=monitor['events'],physical_trace_sha256=hashlib.sha256((folder/'physics-trace.npz').read_bytes()).hexdigest(),
            production_collision_gate_exercised=result['collision_gate_exercised']))
        rows[-1]['max_allocated_finger_force_n']=force_max
        rows[-1]['max_actual_jaw_center_offset_mm']=np.max(abs((actual_fingers[:,[0,2]]-actual_fingers[:,[1,3]])/2),axis=0).tolist()
        rows[-1]['max_actual_jaw_center_offset_mm']=[x*1000 for x in rows[-1]['max_actual_jaw_center_offset_mm']]
    report=dict(data_kind='diagnostic',scope='retimed expert direct-physics ARX reference with shadow contact guard; not policy or production planner acceptance',gpu_used=False,model_used=False,formal_run_lock_frozen=False,
                all_verified=all(r['verification_passed'] for r in rows),task_successes=sum(r['task_success'] for r in rows),episodes=rows)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]/'research/g2');p.add_argument('--output',type=Path,required=True);p.add_argument('--force-limit-n',type=float,nargs='+',default=[2.]);p.add_argument('--episodes',type=int,nargs='+',default=[0,1,2])
    a=p.parse_args();force=a.force_limit_n[0] if len(a.force_limit_n)==1 else a.force_limit_n;verify(a.directory,a.reference,a.source_root,a.output,force,a.episodes)
