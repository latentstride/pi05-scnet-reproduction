"""Verify the real duplicate-input cases now have identical real-command labels."""
import json
from pathlib import Path
import sys
import numpy as np
from scripts.step5_arx.data import Dataset
from unique_observations import relabel

root,mapping_path,aliases_path,output=map(Path,sys.argv[1:])
mapping=json.loads(mapping_path.read_text());aliases=json.loads(aliases_path.read_text())
checked=[]
with Dataset(root/'data/step5-arx/arx-drive-target-v1.h5') as d:
    for rec in aliases:
        split=rec['split'];ep=rec['episode']
        for group in rec['exact_input_conflicts']:
            samples=[relabel(d,mapping,split,ep,t) for t in group['rows']]
            for t,s in zip(group['rows'],samples):
                old=d.read(split,ep,t,terminal=True)
                np.testing.assert_array_equal(s['state'],old['state'])
                for cam in old['images']:np.testing.assert_array_equal(s['images'][cam],old['images'][cam])
                indices=np.asarray(s['target_raw_indices'])
                assert np.all(indices>t) and np.all(np.diff(indices)>=0)
                np.testing.assert_array_equal(s['actions'],d.groups[split][ep]['drive_target'][:][indices])
                np.testing.assert_array_equal(s['actions'],samples[0]['actions'])
            checked.append(dict(split=split,episode=ep,raw_rows=group['rows'],
                                repaired_grip_k4=samples[0]['actions'][:4,6].tolist()))
output.write_text(json.dumps(dict(groups_checked=len(checked),passed=True,
    unchanged_inputs=True,all_targets_from_original_archive=True,groups=checked),indent=2)+'\n')
print(json.dumps(dict(groups_checked=len(checked),passed=True)),flush=True)
