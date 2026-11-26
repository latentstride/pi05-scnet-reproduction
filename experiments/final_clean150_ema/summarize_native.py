"""Read-only common metrics, independent of recording frequency or early stops."""
import json,math
from pathlib import Path
import sys

root=Path(sys.argv[1])
rows=[]
for model in ('native',):
    for seed in sorted(int(p.name.split('-')[-1]) for p in (root/model).glob('seed-*') if p.is_dir()):
        out=root/model/f'seed-{seed}'
        if not (out/'complete.json').exists():
            continue
        result=json.loads((out/'complete.json').read_text())
        setup=json.loads((out/'setup.json').read_text()) if (out/'setup.json').exists() else {}
        dt=setup.get('dt',.004)
        starts=[None,None]
        longest=[0.,0.]
        left_held=False
        right_since=None
        right_longest=0.
        transfer=False
        grasped=False
        maximum=0.
        held_maximum=0.
        stable_maximum=0.
        official=False
        last=None
        events={}
        if (out/'physics.jsonl').exists():
            with (out/'physics.jsonl').open() as stream:
                for line in stream:
                    frame=json.loads(line)
                    last=frame
                    tick=frame['tick']
                    lift=frame['lift_mm']
                    maximum=max(maximum,lift)
                    both=frame['bilateral']
                    if any(both):
                        held_maximum=max(held_maximum,lift)
                    for si in (0,1):
                        if both[si]:
                            if starts[si] is None:
                                starts[si]=tick
                            duration=(tick-starts[si]+1)*dt
                            longest[si]=max(longest[si],duration)
                            if duration>=.2 and not frame['support_contact']:
                                stable_maximum=max(stable_maximum,lift)
                                if lift>=30:
                                    grasped=True
                                    events.setdefault('first_grasp_lift',frame)
                                    if si==0:
                                        left_held=True
                        else:
                            starts[si]=None
                    left_contact=any(c['role'] and c['role'][0]=='left' for c in frame['contacts'])
                    if left_held and both[1] and not left_contact and not frame['support_contact']:
                        if right_since is None:
                            right_since=tick
                        right_longest=max(right_longest,(tick-right_since+1)*dt)
                        if (tick-right_since+1)*dt>=.2:
                            transfer=True
                            events.setdefault('first_right_independent_hold',frame)
                    else:
                        right_since=None
                    official|=frame['native_success']
                    if frame['native_success']:
                        events.setdefault('first_official_success',frame)
        row=dict(model=model,seed=seed,status=result['status'],stop_reason=result.get('stop_reason'),
                 actions=result['actions'],queries=result['queries'],grasped=grasped,
                 max_lift_mm=maximum,max_bilateral_lift_mm=held_maximum,
                 stable_bilateral_max_lift_mm=stable_maximum,longest_bilateral_s=longest,
                 transferred_to_right=transfer,longest_right_independent_s=right_longest,official_check_success=bool(official or result.get('native_success')),
                 placed_observed=result.get('placed',False),video=str(out/'head.mp4'))
        if last is not None:
            delta=[(a-b)*1000 for a,b in zip(last['box_functional_point'],last['target_functional_point'])]
            row.update(final_target_delta_mm=delta,final_horizontal_mm=math.hypot(*delta[:2]),final_vertical_mm=delta[2],
                       final_right_grip=last['drive'][13],final_bilateral=last['bilateral'],final_support=last['support_contact'])
        rows.append(row)
        (out/'read-only-analysis.json').write_text(json.dumps(dict(metrics=row,events=events,last_frame=last),indent=2)+'\n')
(root/'native-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
text=['| 模型 | seed | 抓起≥30mm | 最大净抬升 mm | 稳定双指持物最高 mm | 交接右手 | 官方成功 | 停止原因 |',
      '|---|---:|---|---:|---:|---|---|---|']
for r in rows:
    text.append('| {model} | {seed} | {grasped} | {max_lift_mm:.2f} | {stable_bilateral_max_lift_mm:.2f} | {transferred_to_right} | {official_check_success} | {stop_reason} |'.format(**r))
(root/'native-summary.md').write_text('\n'.join(text)+'\n')
print(json.dumps(rows,indent=2))
