"""Derive a sensitivity estimate from G1 inference and G2 CPU motion evidence."""
import argparse
import csv
import json
from math import ceil
from pathlib import Path

def estimate(root, actions, overhead_s):
    evidence = root / 'g1-reference'
    with (evidence/'queries.csv').open() as stream:
        query_s = max(float(row['inference_s']) for row in csv.DictReader(stream) if row['status']=='ok')
    arx = json.loads((root/'production-gate-20261106-r3/acceptance.json').read_text())['live_result']
    placement = json.loads((root/'placement-20261106/r3/suffix-result.json').read_text())
    arx_per_action_s = (arx['planning_s']+arx['control_s'])/arx['executed']
    scenarios = []
    for pairs in (1,2,3):
        for effective_w4 in (4,2,1):
            counts = [actions,ceil(actions/effective_w4)]
            # Same CPU planning/control cost is used for both windows.
            known = overhead_s + pairs*(2*actions*arx_per_action_s + 2*sum(counts)*query_s)
            scenarios.append(dict(pairs=pairs,effective_window_w4=effective_w4,policy_episodes=pairs*4,
                total_queries=pairs*2*sum(counts),modeled_minutes=known/60,
                historical_cost_scenario_yuan=known*.06/60,
                remaining_minutes_under_historical_20_yuan=(20/.06)-known/60))
    return dict(status='sensitivity_only_not_execution_budget',action_ceiling_assumption=actions,
        startup_overhead_s_assumption=overhead_s,g1_slowest_archived_steady_query_s=query_s,
        cpu_arx_planning_and_control_s_per_completed_action_proxy=arx_per_action_s,
        cpu_arx_proxy_partial_last_action_included=True,
        local_placement_cpu_wall_s=placement['wall_s'],
        historical_rate_yuan_per_minute=.06,historical_ceiling_yuan=20.,
        unmodeled=['Aloha control and rendering','ARX rendering and scene reset per episode',
                   'actual warmup and stage validation','additional expert replay and evidence collection',
                   'GPU execution and memory jointly measured on the selected node'],
        scenarios=scenarios)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('research',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--actions',type=int,default=300)
    p.add_argument('--overhead-s',type=float,default=1200.)
    a=p.parse_args()
    if a.actions<=0 or a.overhead_s<0:
        p.error('actions must be positive and overhead nonnegative')
    a.output.write_text(json.dumps(estimate(a.research,a.actions,a.overhead_s),indent=2)+'\n')
