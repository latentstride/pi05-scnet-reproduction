"""One explicitly authorized G2-E submission; refuses a second attempt."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess

root=Path('<PROJECT_ROOT>')
release=root/'tools/g2e-release-20261106-r1'
digest='fb24a4c727a45945ebe46faaa036f26837c1ab79c3e4d7590433c9f84ef6bfe5'
assert sha256((release/'SHA256SUMS').read_bytes()).hexdigest()==digest
lines=(release/'SHA256SUMS').read_text().splitlines()
assert len(lines)==82
for line in lines:
    h,n=line.split('  ',1)
    assert sha256((release/n).read_bytes()).hexdigest()==h,n
admit=json.loads((release/'release-admission.json').read_text())
assert admit['status']=='ready' and admit['gpu_release_ready'] and admit['cpu_release_validated']
assert admit['approved_allocation_s']==12600
jobs=subprocess.check_output(['squeue','-h','-u','acvkpnrzir','-o','%i %j'],universal_newlines=True)
assert not any(line.split()[-1]=='cpf-pace-g2' for line in jobs.splitlines()),jobs
receipt=root/'logs/.pace-g2e-20261106-r1-once'
receipt.mkdir()  # Atomic, permanent single-attempt marker, even on RPC ambiguity.
(receipt/'queue-before.txt').write_text(jobs)
for p in list(release.rglob('__pycache__')):
    shutil.rmtree(p)
for p in release.rglob('*'):
    p.chmod(0o555 if p.is_dir() else 0o444)
release.chmod(0o555)
argv=['sbatch','--parsable','--time=03:30:00','--time-min=210','--no-requeue',
      '--chdir='+str(root),str(release/'scripts/pace_g2e.slurm'),str(release),digest]
record=dict(authorized_by='2026-11-06 user: 提交作业吧',attempted_at_utc=datetime.now(timezone.utc).isoformat(),
            release=str(release),manifest_sha256=digest,argv=argv,
            approved_allocation_s=12600,one_gpu=True,max_native_trials=8,arx_trials=0,
            verified_price_yuan_per_gpu_hour=3.6,max_cost_yuan=12.6)
(receipt/'submission-attempt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
result=subprocess.run(argv,cwd=str(root),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45)
(receipt/'sbatch.stdout').write_text(result.stdout)
(receipt/'sbatch.stderr').write_text(result.stderr)
if result.returncode or not re.fullmatch(r'\d+(?:;[^\s]+)?',result.stdout.strip()):
    raise SystemExit('Submission response needs read-only reconciliation; do not retry: '+repr(result))
job=result.stdout.strip().split(';')[0]
(receipt/'job-id.txt').write_text(job+'\n')
record.update(job_id=job,returncode=result.returncode)
(receipt/'submission.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('G2E_SUBMITTED_ONCE '+job,flush=True)
