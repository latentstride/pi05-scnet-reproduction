#!/usr/bin/env python3
"""Audit independent-finger state reduction against recorded physics states, CPU only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from g2_assets import load_arms
def save_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','audit','config','recording','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    arms=load_arms(a.assets,a.audit,'ARX-X5',cfg['calibration']['ARX-X5'])
    data=json.loads(a.recording.read_text());states=[];last=-1
    a.output.mkdir(parents=True,exist_ok=False)
    for row in data:
        if row['tick']<=last:raise ValueError('recording ticks must increase')
        last=row['tick'];native=np.array(row['actual_fingers'],dtype=float);original=native.copy()
        mapped=[arm.grip.measure(value) for arm,value in zip(arms,native)]
        if native.shape!=(2,2) or not np.array_equal(native,original):raise ValueError('invalid or mutated measured state')
        states.append(dict(tick=last,grippers=mapped))
    if not states:raise ValueError('empty recording')
    save_json(a.output/'states.json',states)
    save_json(a.output/'acceptance.json',dict(data_kind='diagnostic',gpu_used=False,passed=True,
              source=str(a.recording.resolve()),source_sha256=sha256(a.recording.read_bytes()).hexdigest(),
              config_sha256=sha256(a.config.read_bytes()).hexdigest(),physical_steps_replayed=len(states),
              rejected=0,raw_states_unchanged=True,final=states[-1],
              scope='recording replay of gripper reduction, not a new dynamic run or policy inference'))
