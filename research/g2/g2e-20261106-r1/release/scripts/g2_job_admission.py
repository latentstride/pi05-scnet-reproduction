"""Verify the reviewed release and the running Slurm allocation before graphics."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time


def read_allocation(job_id, out):
    for attempt in range(1, 4):
        try:
            result = subprocess.run(
                ['scontrol', 'show', 'job', '-o', job_id],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, timeout=5,
            )
            (out/'slurm-allocation.txt').write_text(result.stdout)
            fields = dict(word.split('=', 1) for word in result.stdout.split() if '=' in word)
            if fields.get('JobId') != job_id:
                raise SystemExit('Slurm returned a different job ID')
            if fields.get('JobState') not in ('RUNNING', 'CONFIGURING'):
                raise SystemExit(f"Unexpected Slurm job state {fields.get('JobState')}")
            if fields['JobState'] == 'CONFIGURING':
                raise ValueError('Slurm is still configuring the allocation')
            start, end = (datetime.fromisoformat(fields[key]).timestamp()
                          for key in ('StartTime', 'EndTime'))
            return fields, start, end
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyError, ValueError) as exc:
            detail = getattr(exc, 'output', None) or str(exc)
            print(f'Slurm query attempt {attempt}/3: {detail}', flush=True)
            if attempt < 3:
                time.sleep(2)
    raise SystemExit('Slurm did not return the running allocation and its time bounds')


def admit(release, out, job_id):
    admission = json.loads((release/'release-admission.json').read_text())
    if admission['status'] != 'ready' or admission['gpu_release_ready'] is not True:
        raise SystemExit('Release has not passed the submission gate')
    for key, name in [('run_lock_sha256', 'run_lock.json'), ('launch_spec_sha256', 'launch-spec.json')]:
        if sha256((release/name).read_bytes()).hexdigest() != admission[key]:
            raise SystemExit(f'{name} hash mismatch')
    if json.loads((release/'run_lock.json').read_text())['status'] != 'frozen':
        raise SystemExit('Formal run lock is not frozen')
    spec = json.loads((release/'launch-spec.json').read_text())
    budget = admission['approved_allocation_s']
    required = spec['supervisor_timeout_s'] + spec['graphics_timeout_s'] + 60
    if type(budget) is not int or not 60 < required <= budget:
        raise SystemExit('Launch spec exceeds the reviewed allocation cap')
    receipt = Path(spec['model_cpu_receipt'])
    if not receipt.is_file() or sha256(receipt.read_bytes()).hexdigest() != spec['model_cpu_receipt_sha256']:
        raise SystemExit('Model CPU receipt missing or differs from release')

    fields, start, end = read_allocation(job_id, out)
    if fields['NodeList'] != admission['approved_node']:
        raise SystemExit('Allocated node differs from the reviewed node')
    if not start <= time.time() < end or not required <= end - start <= budget:
        raise SystemExit(f'Allocated wall time falls outside the reviewed {required}–{budget}s range')
    (out/'allocation-end-epoch').write_text(f'{end}\n')
    (out/'graphics-timeout-s').write_text(f"{spec['graphics_timeout_s']}\n")
    print(f"Allocation {job_id} admitted on {fields['NodeList']} with {end - start:.0f}s")


if __name__ == '__main__':
    admit(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
