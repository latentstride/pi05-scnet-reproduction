"""Validate all saved episodes and produce an inspectable evaluation report."""
import argparse
import csv
import json
import pathlib

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('output', type=pathlib.Path)
args = p.parse_args()
out = args.output
rows = list(csv.DictReader((out / 'evaluation_results.csv').open()))
assert len(rows) == 15
assert {(int(r['task_id']), int(r['initial_state_id'])) for r in rows} == {(t, i) for t in range(3) for i in range(5)}
for row in rows:
    assert row['termination'] in ('success', 'step_limit') and not row['error']
    steps = int(row['steps'])
    assert 0 < steps <= 220
    with np.load(row['trajectory'], allow_pickle=False) as data:
        assert data['states'].shape == (steps + 1, 8)
        assert data['executed_actions'].shape == (steps, 7)
        assert data['predicted_chunks'].shape[1:] == (10, 7)
        assert len(data['chunk_start_steps']) == int(row['inference_calls'])
        assert all(np.isfinite(data[k]).all() for k in ('states', 'executed_actions', 'predicted_chunks'))
    reader = imageio.get_reader(row['video'])
    count = sum(1 for frame in reader)
    reader.close()
    assert count == steps + 1, (row['video'], count, steps)
summary = json.loads((out / 'summary.json').read_text())
counts = [sum(r['success'] == 'True' for r in rows if int(r['task_id']) == t) for t in range(3)]
assert sum(counts) == summary['successful']
fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(['Task 0', 'Task 1', 'Task 2'], np.asarray(counts) / 5., color='#3b6b93', width=.55)
for bar, n in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .025, '%d/5' % n, ha='center')
ax.set(ylim=(0, 1.14), ylabel='Success rate', title='pi0.5: restricted LIBERO spatial evaluation')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(str(out / 'success_rates.png'), dpi=180)
fig.savefig(str(out / 'success_rates.pdf'))
plt.close(fig)
lines = ['# π₀.₅ LIBERO 受限闭环评测', '',
         '固定官方权重，libero_spatial 任务 0、1、2，各 5 个初始状态，采集与仿真 seed=7。',
         '每轮 10 步等待，每 5 步重新推理，最多 220 个控制步。', '',
         '15 轮已完成并通过轨迹、有限值和视频帧数核验。成功 %d/15，总成功率 %.2f%%。' % (sum(counts), sum(counts) / 15. * 100), '',
         '|任务|成功数|成功率|', '|---|---:|---:|']
for t, n in enumerate(counts):
    lines.append('|%d|%d/5|%.1f%%|' % (t, n, n / 5. * 100))
lines += ['', '|任务|初始状态|结果|终止原因|控制步数|', '|---|---|---|---|---:|']
for r in rows:
    lines.append('|%s|%s|%s|%s|%s|' % (r['task_id'], r['initial_state_id'], r['success'], r['termination'], r['steps']))
lines += ['', '## 归档与解释', '',
          '`evaluation_results.csv` 保存逐轮结果，`episodes/*.npz` 保存逐步状态、实际执行动作、预测动作块和耗时，`videos/*.mp4` 保存完整控制过程。',
          '状态数比执行动作数多一，可用于检查每次动作前后的状态变化。`chunk_start_steps` 给出每个预测块对应的控制步。',
          '成功判据沿用官方客户端的环境 done；达到 220 步且未成功记录为 step_limit。运行异常单独记录，不能当作有效失败轮次掩盖。',
          '每个任务仅 5 轮，结果属于功能性基线，不能作为官方完整 benchmark 的复现分数。失败轮次仅说明在给定步数内未成功，原因须结合视频分析。',
          '视频以 10 fps 编码，控制频率为 20 Hz，因此回放为半速。', '']
(out / 'evaluation_report.md').write_text('\n'.join(lines))
(out / 'artifact_validation.json').write_text(json.dumps({'validated_episodes': 15, 'finite_trajectories': True, 'video_frame_counts_match': True}, indent=2))
print('ALL_15_EPISODES_AND_VIDEOS_VALIDATED', json.dumps(summary), flush=True)
