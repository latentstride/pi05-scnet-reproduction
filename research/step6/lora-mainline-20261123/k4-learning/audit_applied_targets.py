"""Report controller target changes separately from the native LoRA result."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(trial, safety):
    changed=[]
    approved=0
    with (trial/'trajectory.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['kind']!='approved_row':continue
            approved+=1
            requested=np.asarray(row['input']).reshape(2,7)
            fingers=[]
            for i,side in enumerate(('left','right')):
                grip=safety['calibration'][side]
                closed=np.asarray(grip['closed_m']);opened=np.asarray(grip['opened_m'])
                fingers.append(closed+requested[i,6]*(opened-closed))
            arm_delta=float(np.max(abs(np.asarray(row['joint_goal'])-requested[:,:6])))
            grip_delta=float(np.max(abs(np.asarray(row['finger_goal_m'])-fingers)))
            if arm_delta>1e-8 or grip_delta>1e-8:
                changed.append(dict(action_index=row['action_index'],query_index=row['query_index'],
                    max_arm_change_rad=arm_delta,max_finger_change_m=grip_delta,phase=row.get('phase')))
    return dict(trial=str(trial),approved_rows=approved,changed_targets=changed,
                all_approved_targets_preserve_model_commands=approved>0 and not changed,
                scope='approved arm and finger endpoints; not a claim of task success')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('safety',type=Path);p.add_argument('trials',nargs='+',type=Path)
    a=p.parse_args();safety=json.loads(a.safety.read_text())
    print(json.dumps([audit(t,safety) for t in a.trials],indent=2))
