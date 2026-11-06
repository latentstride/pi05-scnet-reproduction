"""Validate measured object matching; retain workspace exclusions for CPU candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
from g2_initial import compare_objects

def validate_pair(record):
    a,b=(record['snapshots'][k] for k in ('aloha-agilex','ARX-X5'))
    failures=['object_'+n for n in compare_objects(a['initial'],b['initial'])]
    for robot,state in record['snapshots'].items():
        tail=np.asarray(state['last_200_position_range_m'],float)
        velocity=np.asarray(state['block_velocity_m_s'],float)
        angular=np.asarray(state['block_angular_velocity_rad_s'],float)
        if not all(np.isfinite(v).all() for v in (tail,velocity,angular)):
            failures.append(robot+':nonfinite')
        elif max(tail)>1e-5 or np.linalg.norm(velocity)>1e-4 or np.linalg.norm(angular)>1e-3:
            failures.append(robot+':unstable_block')
        box_contacts=[c for c in state['contacts'] if 'box' in c['actors']]
        if not box_contacts or any(set(c['actors'])!={'box','table'} for c in box_contacts):
            failures.append(robot+':unexpected_block_support')
        if state['physics_steps_after_spawn']!=2500:
            failures.append(robot+':settling_steps')
    return failures

def build(data, initial_collision=None, path_reports=None):
    candidates=[]
    for row in data['records']:
        failures=validate_pair(row)
        reasons=['full_task_common_workspace_not_checked','initial_robot_self_contacts_require_review',
                 'original_expert_actual_robot_initial_state_not_recovered',
                 'CPU_explicit_opened_home_initialization_differs_from_source_gripper_ramp']
        geometry_pass=False
        if initial_collision is not None:
            checks=[v for v in initial_collision if v['episode']==row['episode']]
            if len(checks)!=2 or {v['robot'] for v in checks}!={'aloha-agilex','ARX-X5'}:
                raise ValueError('missing embodiment collision evidence')
            for v in checks:
                measured=row['snapshots'][v['robot']]['initial']['actual_arm_qpos_rad']
                if not np.array_equal(v['actual_arm_qpos_rad'],[measured[s] for s in ('left','right')]):
                    raise ValueError('collision evidence uses a different actual state')
            geometry_pass=all(v['fcl_clear'] and not v['blocked_collisions'] for v in checks)
            if geometry_pass:reasons.remove('initial_robot_self_contacts_require_review')
        path=None
        if path_reports is not None:
            path=next(v for v in path_reports if v['episode']==row['episode'])
            reasons.remove('full_task_common_workspace_not_checked')
            reasons.append('held_object_and_contact_path_unverified')
            if path['segments_with_known_collision']:
                reasons.append('selected_expert_path_hits_target_tray')
        candidates.append(dict(candidate_id='official-expert-episode'+str(row['episode'])+'-seed'+str(row['seed']),
            formal_pair_id=None,episode=row['episode'],seed=row['seed'],data_kind='diagnostic',
            object_pair_check_passed=not failures,object_pair_failures=failures,
            source_kind='actual_CPU_reconstruction_after_settling',
            actual_initial={r:s['initial'] for r,s in row['snapshots'].items()},
            initial_fcl_geometry_passed=geometry_pass,
            known_colliding_path_segments=None if path is None else path['segments_with_known_collision'],
            formal_exclusion_reasons=reasons))
    return dict(schema_version=1,formal_roster_frozen=False,candidates=candidates)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collision-directory',type=Path)
    a=p.parse_args();extra={}
    if a.collision_directory:
        extra=dict(initial_collision=json.loads((a.collision_directory/'initial-collision.json').read_text()),
                   path_reports=[v for f in sorted(a.collision_directory.glob('workspace-collision-episode*.json'))
                                 if (v:=json.loads(f.read_text())).get('robot','ARX-X5')=='ARX-X5'])
    result=build(json.loads(a.input.read_text()),**extra);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print([(r['candidate_id'],r['object_pair_check_passed']) for r in result['candidates']])
