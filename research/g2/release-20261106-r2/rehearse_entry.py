"""Drive the exact batch and supervisor scripts against isolated CPU fixtures."""
from datetime import datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def executable(path, source):
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + source)
    path.chmod(0o700)


def rehearse(source, output):
    with tempfile.TemporaryDirectory(prefix='g2-entry-', dir=output) as scratch:
        home = Path(scratch)
        root = home/'code_cpf/pi05'
        release = root/'tools/rehearsal-only'
        scripts = release/'scripts'
        scripts.mkdir(parents=True)
        (root/'results').mkdir()
        fixtures = home/'bin'
        fixtures.mkdir()
        copied = (
            'pace_g2.slurm', 'run_lock.json',
            'scripts/g2_job_admission.py', 'scripts/g2_single_launch.py',
            'scripts/run-admitted-g2-plan.sh', 'scripts/g2_host_supervisor.py',
            'scripts/pace_g1_orchestrate.py', 'scripts/g2_single_stage.py',
        )
        for name in copied:
            shutil.copyfile(source/name, release/name)

        spec = json.loads((source/'launch-spec.json').read_text())
        receipt = home/'cpu-receipt.json'
        receipt.write_text('{"scope":"synthetic receipt, rehearsal only"}\n')
        spec.update(model_cpu_receipt=str(receipt), model_cpu_receipt_sha256=digest(receipt))
        (release/'launch-spec.json').write_text(json.dumps(spec))
        admission = dict(status='ready', gpu_release_ready=True, approved_node='gnode39',
                         approved_allocation_s=18000, run_lock_sha256=digest(release/'run_lock.json'),
                         launch_spec_sha256=digest(release/'launch-spec.json'))
        (release/'release-admission.json').write_text(json.dumps(admission))

        executable(fixtures/'scontrol', '''
[[ "$*" == "show job -o $SLURM_JOB_ID" ]]
if [[ "$CASE" == transient && ! -f "$ATTEMPT" ]]; then
    touch "$ATTEMPT"
    echo "synthetic controller connection failure" >&2
    exit 1
fi
printf '%s\n' "$JOB_ROW"
''')
        executable(fixtures/'nvidia-smi', '''
printf 'synthetic, GPU-fixture, 14000, 24576\n'
exec sleep 30
''')
        executable(fixtures/'sim-only', '''
echo "synthetic stage $*" >> "$STAGE_CALLS"
''')
        executable(scripts/'g2_graphics.sh', '''
mkdir "$3"
if [[ "$CASE" == graphics_failure ]]; then
    echo "graphics_driver_discovery exit=15" > "$3/FAILED"
    exit 15
fi
printf '{"gpu_uuid":"GPU-fixture","sim_argv":["%s"]}\n' "$SIM_FIXTURE" > "$3/sim-launch.json"
''')
        executable(scripts/'pace_g1_model_env.sh', '''
if [[ "$CASE" == model_failure ]]; then
    echo "synthetic model startup failure" >&2
    exit 31
fi
[[ "$3" == --port ]]
printf 'checkpoint policy on ws://127.0.0.1:%s\n' "$4"
exec sleep 30
''')
        manifest = release/'SHA256SUMS'
        manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(release)}\n'
                                    for p in sorted(release.rglob('*')) if p.is_file()))
        baseline = receipt.read_text()
        cases = [
            ('happy', 0, 'supervisor'),
            ('transient', 0, 'supervisor'),
            ('manifest_mismatch', 1, 'release_manifest'),
            ('receipt_mismatch', 1, 'allocation'),
            ('wrong_node', 1, 'allocation'),
            ('short_wall', 1, 'allocation'),
            ('long_wall', 1, 'allocation'),
            ('graphics_failure', 15, 'graphics'),
            ('model_failure', 1, 'supervisor'),
        ]
        checks = {}
        for i, (name, code, stage) in enumerate(cases, 100):
            job = str(i)
            duration = {'short_wall': 600, 'long_wall': 18001}.get(name, 18000)
            start = datetime.now().replace(microsecond=0) - timedelta(seconds=1)
            end = start + timedelta(seconds=duration)
            node = 'gnode65' if name == 'wrong_node' else 'gnode39'
            row = (f'JobId={job} JobState=RUNNING NodeList={node} '
                   f'StartTime={start.isoformat()} EndTime={end.isoformat()}')
            supplied = '0'*64 if name == 'manifest_mismatch' else digest(manifest)
            receipt.write_text(baseline if name != 'receipt_mismatch' else 'changed')
            calls = output/f'{name}-stage-calls.log'
            env = dict(os.environ, HOME=str(home), PATH=f'{fixtures}:{os.environ["PATH"]}',
                       SLURM_JOB_ID=job, SLURM_JOB_NAME='cpf-pace-g2', SLURM_SUBMIT_DIR=str(root),
                       CASE=name, JOB_ROW=row, ATTEMPT=str(home/'transient-seen'),
                       SIM_FIXTURE=str(fixtures/'sim-only'), STAGE_CALLS=str(calls),
                       PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run(['bash', str(release/'pace_g2.slurm'), str(release), supplied],
                                    cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True, timeout=20)
            (output/f'{name}.log').write_text(result.stdout)
            run = root/'results'/f'pace-g2-{job}'
            status = (run/'entry-status.tsv').read_text()
            (output/f'{name}-entry-status.tsv').write_text(status)
            if result.returncode != code:
                raise AssertionError(f'{name}: exit {result.returncode}, expected {code}\n{result.stdout}')
            assert status.strip().endswith(f'{stage}\t' + (f'failed exit={code}' if code else 'complete')), name
            if code:
                assert (run/'entry-exit-code').read_text().strip() == str(code), name
                assert not calls.exists(), name
            if name == 'transient':
                assert 'Slurm query attempt 1/3' in (run/'allocation-admission.log').read_text()
            if stage in ('release_manifest', 'allocation'):
                assert not (run/'graphics').exists(), name
            if name == 'graphics_failure':
                assert (run/'graphics/FAILED').read_text().strip() == 'graphics_driver_discovery exit=15'
                assert not (run/'supervisor').exists()
            if stage == 'supervisor':
                report = json.loads((run/'supervisor/g2_acceptance.json').read_text())
                assert report['complete'] == (code == 0), name
                assert report['model_server_starts'] == 1, name
                if code:
                    assert 'exited before ready' in report['model_start_error']
                    assert report['stages']['aloha']['status'] == 'skipped'
                else:
                    assert len(calls.read_text().splitlines()) == 12
            checks[name] = dict(exit_code=code, last_stage=stage)
        return dict(passed=True, gpu_used=False, fixtures=['Slurm controller', 'graphics', 'model', 'GPU telemetry', 'simulator'],
                    tested_source={name: digest(source/name) for name in copied}, checks=checks)


if __name__ == '__main__':
    source, output = (Path(x).resolve() for x in sys.argv[1:])
    output.mkdir()
    report = rehearse(source, output)
    (output/'entry-rehearsal.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"ENTRY_REHEARSAL_PASS {len(report['checks'])}")
