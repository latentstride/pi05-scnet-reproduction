"""Summarize archived production timing by physical contact phase."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

def summarize(directory, control_samples=None):
    result = json.loads((directory / 'result.json').read_text())
    groups = defaultdict(lambda: dict(wall_s=0., control_interval_s=0., physics_steps=0,
                                      control_calls=0, planning_calls=0, control_samples=[]))
    phase = 'approach'
    previous = None
    current_call_s = None
    call_rows = []
    def record_call(bounds):
        start, last, owner = bounds
        if last > start:
            groups[owner]['control_samples'].append(last - start)
            call_rows.append((len(call_rows)+1, owner, start, last, last-start))
    for line in (directory / 'events.jsonl').open():
        row = json.loads(line)
        wall = row['elapsed_s']
        phase = (row.get('contact_state') or {}).get('phase', phase)
        group = groups[phase]
        if previous:
            group['wall_s'] += wall - previous['elapsed_s']
            if row['phase'] == previous['phase'] == 'control':
                group['control_interval_s'] += wall - previous['elapsed_s']
        if row.get('kind') == 'expert_window':
            group['planning_calls'] += 1
        group['physics_steps'] += row.get('physics_step_count', 0)
        if row.get('control_call_count'):
            group['control_calls'] += row['control_call_count']
            if current_call_s is not None:
                record_call(current_call_s)
            current_call_s = (previous['elapsed_s'] if previous else wall, wall, phase)
        elif row.get('physics_step_count') and current_call_s is not None:
            current_call_s = (current_call_s[0], wall, current_call_s[2])
        previous = row
    if current_call_s is not None:
        record_call(current_call_s)
    if control_samples is not None:
        with control_samples.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('call','contact_phase','event_before_s','last_step_event_s','estimated_wall_s'))
            writer.writerows(call_rows)
    for group in groups.values():
        samples = group.pop('control_samples')
        group['control_calls_with_timing'] = len(samples)
        group['mean_control_wall_s'] = mean(samples) if samples else None
        group['median_control_wall_s'] = median(samples) if samples else None
        group['sim_s'] = group['physics_steps'] * .004
        group['sim_s_per_wall_s'] = group['sim_s'] / group['wall_s']
    return dict(scope='phase by measured holder; elapsed intervals assigned to next event; per-call times from event bounds',
                totals={k: result[k] for k in ('elapsed_s','planning_s','control_s','plan_calls','control_calls','outcome')},
                phases=dict(groups))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--control-samples',type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(summarize(args.directory,args.control_samples), indent=2)+'\n')
