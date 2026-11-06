"""Exercise the packaged host launch plan without making a resource allocation."""
import argparse
import json
from pathlib import Path
from time import time

from g2_host_supervisor import validate_plan
from g2_single_launch import build


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('release','root','output'):
        p.add_argument(n,type=Path)
    a=p.parse_args()
    a.output.mkdir(exist_ok=True,parents=True)
    spec=json.loads((a.release/'launch-spec.json').read_text())
    fake_gpu=dict(gpu_uuid='GPU-00000000-0000-0000-0000-000000000000',
                  sim_argv=['cpu-preflight-placeholder-no-launch'])
    plan=build(a.root,a.release,a.output,'1',fake_gpu,time()+spec['approved_allocation_s'])
    validate_plan(plan)
    stage_counts={}
    for name in plan['stages']:
        commands=json.loads((a.output/f'{name}-commands.json').read_text())
        stage_counts[name]=len(commands)
        if not all(command[:1]==fake_gpu['sim_argv'] for command in commands):
            raise ValueError('stage command bypassed the audited simulator vector')
    if stage_counts!={'dynamic':2,'aloha':2,'arx':2,'ablation':4,'expert':1,'statistics':1}:
        raise ValueError(f'incomplete stage roster: {stage_counts}')
    expected=sum(spec['stage_timeout_s'].values())+spec['graphics_timeout_s']+60
    if expected>spec['approved_allocation_s']:
        raise ValueError('sum of stage hard caps exceeds allocation')
    (a.output/'host-plan-check.json').write_text(json.dumps(dict(passed=True,no_process_launched=True,
        commands_per_stage=stage_counts,bounded_s=expected,allocation_s=spec['approved_allocation_s']),indent=2)+'\n')
