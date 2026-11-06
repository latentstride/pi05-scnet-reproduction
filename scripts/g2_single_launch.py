"""Build in-allocation G2 commands from fixed paths, lock and the graphics probe."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

PACE = 'fd2e624cc19325cc9b20a620c82d51c0d8e25cc1'
OPENPI = 'f4e52e6de4cc487a11585a3a7414d610b2883dae'


def build(root, release, output, job, graphics, allocation_end):
    spec = json.loads((release/'launch-spec.json').read_text())
    lock = release/'run_lock.json'
    roster = json.loads(lock.read_text())['episodes']
    digest = sha256(lock.read_bytes()).hexdigest()
    gpu = graphics['gpu_uuid']
    if not gpu.startswith('GPU-'):
        raise ValueError('missing allocation GPU UUID')
    pace = root/'source'/f'PACE-{PACE}'
    openpi = root/'source'/f'Openpi-PACE-{OPENPI}'
    pythonpath = ':'.join(map(str,(release/'scripts',root/'tools/pace-smoke-deps',openpi/'packages/openpi-client/src')))
    sim = graphics['sim_argv'] + ['env', f'SLURM_JOB_ID={job}', 'SLURM_JOB_NAME=cpf-pace-g2',
                                  f'PYTHONPATH={pythonpath}', 'PYTHONFAULTHANDLER=1', 'MPLBACKEND=Agg']
    py = sim + ['python']
    port = 50000 + int(job)%14000
    uri = f'ws://127.0.0.1:{port}'
    receipt = Path(spec['model_cpu_receipt'])
    if sha256(receipt.read_bytes()).hexdigest() != spec['model_cpu_receipt_sha256']:
        raise ValueError('model CPU receipt differs from release')
    model_dir = output/'model'
    model = ['env',f'PI05_ROOT={root}','bash',str(release/'scripts/pace_g1_model_env.sh'),gpu,
             str(release/'scripts/pace_model_smoke.py'),'--port',str(port),'--require-gpu',
             '--openpi-root',str(openpi),'--download-dir',str(root/'cache/pace-handover'),
             '--diagnostics-dir',str(model_dir),'--verified-model',str(receipt)]
    stages = {}

    def trial(row, startup=False):
        episode = row['episode_id']
        dest = output/('startup' if startup else 'trials')/episode
        command = py + [str(release/'scripts/g2_locked_task.py'),'--lock',str(lock),'--lock-sha',digest,
            '--episode-id',episode,'--config',str(release/'configs'/f'{episode}.json'),
            '--pace-root',str(pace),'--source-audit',str(release/'source-audit.json'),
            '--asset-audit',str(release/'asset-audit.json'),'--model-manifest',str(release/'pace_download_manifest.json'),
            '--model-startup',str(model_dir/'startup.json'),'--output',str(dest),'--uri',uri,
            '--policy-timeout-s','390' if startup else '90']
        return command + (['--startup-timeout-s','480' if row['condition']['embodiment']=='aloha-agilex' else '280'] if startup else [])

    stages['dynamic'] = dict(commands=[trial(r,True) for r in roster if r['seed']==0 and r['condition']['window_actions']==1],
                             requires=[],needs_model=True)
    for stage, condition in [('aloha','A-native-w1'),('arx','B-arx-w1'),('ablation',None)]:
        chosen = [r for r in roster if (r['condition']['id']==condition if condition else r['condition']['window_actions']==4)]
        stages[stage] = dict(commands=[trial(r) for r in chosen],requires=['dynamic'],needs_model=True)
    expert = output/'expert-reference'
    stages['expert'] = dict(commands=[py+[str(release/'scripts/g2_arx_expert_monitor_cpu_v5.py'),
        '--assets',str(pace/'assets'),'--audit',str(release/'asset-audit.json'),
        '--config',str(release/'configs/seed0-B-arx-w1.json'),'--input',str(release/'expert/input.json'),
        '--trace',str(release/'expert/trace.npz'),'--reference',str(release/'expert/reference-q.npz'),
        '--output',str(expert)]],requires=['dynamic'],needs_model=False)
    stages['statistics'] = dict(commands=[py+[str(release/'scripts/g2_single_statistics.py'),
        str(lock),digest,str(output/'trials'),str(output/'statistics'),str(expert/'replay.json')]],
        requires=['aloha','arx','ablation','expert'],needs_model=False)
    # JSON stage scripts contain only argv vectors. Use the host Python to avoid nested containers.
    for name,row in stages.items():
        batch = output/f'{name}-commands.json'
        batch.write_text(json.dumps(row.pop('commands'),indent=2)+'\n')
        row['argv']=['python3',str(release/'scripts/g2_single_stage.py'),str(batch)]
        if name=='dynamic':
            row['argv'] += ['--monitor',str(output/'supervisor/gpu-memory.csv')]
        row['timeout_s']=spec['stage_timeout_s'][name]
    return dict(data_kind='diagnostic',total_timeout_s=spec['supervisor_timeout_s'],deadline_epoch=allocation_end-60,
        model_ready_timeout_s=960,model_server=model,ready_line=f'checkpoint policy on {uri}',
        model_phases_file=str(model_dir/'phases.jsonl'),
        model_phase_timeouts=dict(container_startup=180,checkpoint_and_tokenizer_sha256=180,
            openpi_import=480,jax_device=120,orbax_restore_and_policy_create=600),
        monitor=['nvidia-smi',f'--id={gpu}','--query-gpu=timestamp,uuid,memory.used,memory.total',
                 '--format=csv,noheader,nounits','--loop=1'],stages=stages)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','release','output','graphics'):
        p.add_argument(name,type=Path)
    p.add_argument('job')
    p.add_argument('allocation_end',type=float)
    a=p.parse_args()
    plan=build(a.root,a.release,a.output,a.job,json.loads(a.graphics.read_text()),a.allocation_end)
    (a.output/'effective-launch-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
