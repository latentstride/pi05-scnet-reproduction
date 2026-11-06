"""Check reconstructed reset states and the first production ARX observation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sapien

from g2_episode import save_json
from g2_gate_scene_cpu import build
from g2_initial import apply_fixture, capture_initial, compare_objects, digest_json

def run(args):
    rows=json.loads(args.candidates.read_text())['candidates']
    reports=[]
    for row in rows:
        episode=row['episode']; fixture=row['fixture']
        scene=build(args.assets,args.audit,args.config,args.inputs/f'episode{episode}.json',
                    args.reference,args.output/f'episode{episode}')
        actors=dict(box=scene.block,target_box=scene.target,
                    table=next(e for e in scene.scene.get_all_actors() if e.name=='table'))
        proxy=scene.adapter.source_collision.refresh.__self__
        aloha_robot=SimpleNamespace(left_entity=proxy.entity,right_entity=proxy.entity)
        arms_by_robot={'aloha-agilex':scene.adapter.aloha,'ARX-X5':scene.arms}
        robots={'aloha-agilex':aloha_robot,'ARX-X5':scene.robot}
        envs={k:SimpleNamespace(robot=robot,**actors) for k,robot in robots.items()}
        snaps={}; checks={}
        for robot in ('aloha-agilex','ARX-X5'):
            arms=arms_by_robot[robot]
            if robot=='aloha-agilex':
                from dataclasses import replace
                from g2_capture import articulation_id
                arms=tuple(replace(arm,instance=articulation_id(proxy.entity)) for arm in arms)
            env=envs[robot]
            apply_fixture(env,arms,fixture,robot,sapien.Pose)
            snap=capture_initial(env,arms,np.eye(4),str(episode),digest_json(fixture),'10g-reconstructed-reset')
            snaps[robot]=snap
            target=fixture['robots'][robot]
            q=np.array([snap['actual_arm_qpos_rad'][side] for side in ('left','right')])
            g=np.array([target[side]['fingers_m'] for side in ('left','right')])
            error=np.max(np.abs(q-np.array([target[side]['q_rad'] for side in ('left','right')])))
            collision=proxy.checker if robot=='aloha-agilex' else scene.contact
            proxy.refresh(); scene.contact.refresh()
            checks[robot]=dict(clear=bool(collision(q,g)),blocked=collision.last_collisions,
                               joint_readback_max_rad=float(error))
        scene.backend.episode=f'pair-{episode}'
        snap,_=scene.backend.capture()
        observation=scene.adapter.observe(snap,'Hand the block to the other arm and place it on the target.',45.)
        mismatches=compare_objects(snaps['aloha-agilex'],snaps['ARX-X5'])
        report=dict(episode=episode,fixture_sha256=digest_json(fixture),objects_mismatched=mismatches,
                    actual_initial=snaps,geometry=checks,arx_first_observation=dict(ok=observation.ok,
                    status=observation.status.value,detail=observation.detail,diagnostic=observation.diagnostic))
        reports.append(report)
        save_json(args.output/'pair-probe.json',reports)
        print('PAIR',episode,'objects',mismatches,'clear',[v['clear'] for v in checks.values()],
              'observation',observation.status.value,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','audit','config','inputs','reference','candidates','output'):
        p.add_argument('--'+name,required=True,type=Path)
    run(p.parse_args())
