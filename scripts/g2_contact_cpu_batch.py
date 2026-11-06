"""Bounded diagnostic force sweep; preserves failures as observations, never formal scores."""
import argparse
import gc
import json
from pathlib import Path
from time import monotonic
import traceback
from g2_contact_cpu_smoke import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','audit','config','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for name,dynamic,force in [('static-final',False,2.),('dynamic-2N',True,2.),
                               ('dynamic-8N',True,8.),('dynamic-12N',True,12.)]:
        start=monotonic();row=dict(condition=name,dynamic=dynamic,force_limit_n=force)
        try:
            row['checks_passed']=run(a.assets,a.audit,a.config,a.output/name,dynamic,force)
        except Exception as error:
            row.update(checks_passed=False,error=repr(error),traceback=traceback.format_exc())
        row['wall_s']=monotonic()-start;records.append(row)
        (a.output/'batch-status.json').write_text(json.dumps(dict(data_kind='diagnostic',gpu_used=False,
              renderer_created=False,completed=len(records)==4,conditions=records),indent=2)+'\n')
        gc.collect()
    print('CONTACT_CPU_BATCH_COMPLETE')
    raise SystemExit(0 if records[0]['checks_passed'] else 1)
