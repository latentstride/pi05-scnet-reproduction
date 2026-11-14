"""Reuse the accepted 100-trajectory data; do not treat frames as physical time."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'research/robotwin100_course/data/dashboard_data.json'


def main():
    data=json.loads(SOURCE.read_text())
    result=dict(source='research/robotwin100_course/data/dashboard_data.json',
        sha256=sha256(SOURCE.read_bytes()).hexdigest(),
        scope='accepted rounded display arrays; command/stage reference only; no new FK or physical replay',
        object_poses_available=False,same_episode_does_not_establish_same_initial_state=True,robots={})
    for robot in ('aloha-agilex','arx-x5'):
        records=[]
        for t in data['trajectories']:
            if t['robot']!=robot:continue
            x=np.asarray(t['state14']);g=x[:,[6,13]]
            assert len(x)==t['n']
            phases={
                'left_closed_right_open':(g[:,0]<.1)&(g[:,1]>.9),
                'both_closed':(g[:,0]<.1)&(g[:,1]<.1),
                'left_open_right_closed':(g[:,0]>.9)&(g[:,1]<.1),
            }
            first=lambda mask: int(np.flatnonzero(mask)[0]) if mask.any() else None
            records.append(dict(episode=t['episode'],frames=t['n'],original_hdf5_sha256=t['source_sha256'],
                first_half_closed_frame=[first(g[:,i]<=.5) for i in range(2)],
                first_nearly_closed_frame=[first(g[:,i]<=.1) for i in range(2)],
                first_phase_frame={k:first(v) for k,v in phases.items()},
                initial_native_joint_command=x[0].tolist()))
        assert len(records)==50
        result['robots'][robot]=dict(trajectories=records,
            phase_present_counts={k:sum(r['first_phase_frame'][k] is not None for r in records)
                                  for k in phases},
            half_closure_median_frame=[float(np.median([r['first_half_closed_frame'][i] for r in records]))
                                       for i in range(2)])
    out=Path(__file__).with_name('expert-comparison.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:{'phases':v['phase_present_counts'],'half_closure_median_frame':v['half_closure_median_frame']}
           for k,v in result['robots'].items()})


if __name__=='__main__':main()
