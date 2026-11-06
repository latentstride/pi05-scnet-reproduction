"""Validate the locked policy records and derive the paired window contrasts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from g2_locked_score import collect
from summarize_experiments import paired_tables

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('lock',type=Path)
    p.add_argument('digest')
    p.add_argument('trials',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('expert',type=Path)
    a=p.parse_args()
    a.output.mkdir()
    records=a.output/'records.json'
    rows=collect(a.lock,a.digest,a.trials,records)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_experiments.py')),
        str(records),'--run-lock',str(a.lock),'--data-kind','formal','--output',str(a.output/'transfer')],check=True)
    ablations=[]
    for embodiment in ('aloha-agilex','arx-x5'):
        ablations.extend(dict(row,comparison_id='window-'+embodiment) for row in rows
                         if row['condition']['embodiment']==embodiment)
    summary, pairs=paired_tables(ablations)
    (a.output/'window-contrasts.json').write_text(json.dumps(dict(pairs=pairs,summary=summary),indent=2)+'\n')
    expert=json.loads(a.expert.read_text())
    passed=(expert['completed'] and expert['contact_monitor_passed'] and
            expert['native_success_at_expert_end']['success'] and expert['native_success_after_diagnostic_hold']['success'])
    (a.output/'expert-reference-check.json').write_text(json.dumps(dict(passed=passed,expert=expert),indent=2)+'\n')
    if not passed:
        raise SystemExit('expert diagnostic reference failed')
