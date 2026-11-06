#!/usr/bin/env python3
"""Audit fixed official expert targets; reconstruct (never invent raw) time/poses."""
import argparse
import hashlib
import io
import json
import pickle
from pathlib import Path
import h5py
import numpy as np

class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError('unexpected global: ' + module + '.' + name)
        return allowed[module, name]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def timeline(paths, initial):
    q = initial.copy()
    rows, ticks, segments = [], [], []
    tick = 0
    stages = [('L',0),('L',1),('LG',0),('L',2),('L',3),('L',4),
              ('R',0),('R',1),('RG',0),('LG',1),('L',5),('B',(6,2)),('R',3),('RG',1)]
    def capture():
        rows.append(q.copy()); ticks.append(tick)
    for kind, index in stages:
        start = len(rows); begin = tick; capture()
        if kind == 'B':
            lp = paths['left_joint_path'][index[0]]['position']
            rp = paths['right_joint_path'][index[1]]['position']
            li = ri = i = 0
            while li < len(lp) or ri < len(rp):
                if li < len(lp) and li / len(lp) <= ri / len(rp):
                    q[:6] = lp[li]; li += 1
                if ri < len(rp) and ri / len(rp) <= li / len(lp):
                    q[7:13] = rp[ri]; ri += 1
                tick += 1
                if i % 15 == 0: capture()
                i += 1
        else:
            if kind in ('L','R'):
                values = paths['left_joint_path' if kind == 'L' else 'right_joint_path'][index]['position']
                dest = slice(0,6) if kind == 'L' else slice(7,13)
            else:
                dest = 6 if kind == 'LG' else 13
                values = np.r_[np.linspace(q[dest], index, 200), np.repeat(index, 100)]
            for i, value in enumerate(values):
                q[dest] = value; tick += 1
                if i % 15 == 0: capture()
        capture()
        segments.append(dict(kind=kind, path_index=index, first_frame=start,
                             frame_count=len(rows)-start, first_tick=begin, last_tick=tick))
    return np.array(rows), np.array(ticks), segments

def spawn(seed, cameras=2):
    # Fixed clean config: table-height draw, then one randn(3)+uniform per static camera.
    rng = np.random.RandomState(seed)
    rng.uniform(0,0)
    for _ in range(cameras):
        v = rng.randn(3); v /= np.linalg.norm(v); rng.uniform(0,0)
    pos = [rng.uniform(-.25,-.05), rng.uniform(0,.25), rng.uniform(.842,.842)]
    angles = [rng.uniform(0,0), rng.uniform(0,0), rng.uniform(-.2,.2)]
    base = np.array([.981,0,0,.195]); base /= np.linalg.norm(base)
    c,s = np.cos(angles[2]/2), np.sin(angles[2]/2)
    quat = [base[0]*c-base[3]*s, 0.,0.,base[0]*s+base[3]*c]
    target = [rng.uniform(.1,.25),rng.uniform(.15,.2),rng.uniform(.741,.741)]
    return dict(box=dict(position_m=pos, quaternion_wxyz=quat),
                target_box=dict(position_m=target, quaternion_wxyz=[1,0,0,0]))

def run(root, cache, output):
    output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((cache/'provenance.json').read_text())
    members = {m['file']:m['sha256'] for m in provenance['members']}
    metadata = root/'research/robotwin/data_probe/aloha-agilex/metadata'
    seeds = list(map(int,(metadata/'seed.txt').read_text().split()))
    report = dict(schema_version=1, time_semantics='source_reconstructed_simulation_time_not_raw_timestamp',
                  generator_revision_proven=False, physics_dt_s=.004, interior_frame_dt_s=.060,
                  source_revision='ea8b21121ebb3cd201ff5b3fe361944ac94eda3f',
                  dataset_revision=provenance['revision'], original_object_pose_snapshot_found=False, episodes=[])
    for episode in range(3):
        file = cache/f'episode{episode}.pkl'
        if sha(file) != members[file.name]: raise ValueError('planner member hash mismatch')
        with file.open('rb') as stream: paths = NumpyOnlyUnpickler(stream).load()
        if set(paths) != {'left_joint_path','right_joint_path'}: raise ValueError('unexpected cache schema')
        if len(paths['left_joint_path']) != 7 or len(paths['right_joint_path']) != 4: raise ValueError('unexpected task stages')
        for p in paths['left_joint_path'] + paths['right_joint_path']:
            if p['status'] != 'Success' or p['position'].ndim != 2 or p['position'].shape[1] != 6 or not np.isfinite(p['position']).all():
                raise ValueError('invalid cached path')
        h5 = root/f'research/robotwin/data_probe/aloha-agilex/episode{episode}.hdf5'
        with h5py.File(h5) as f:
            observed = np.asarray(f['joint_action/vector'])
            predicted, ticks, stages = timeline(paths, observed[0])
            if predicted.shape != observed.shape: raise ValueError('frame count mismatch')
            error = float(np.max(abs(predicted-observed)))
            if error > 1e-12: raise ValueError('target mismatch: '+str(error))
            row = dict(episode=episode, seed=seeds[episode], frames=len(ticks), duration_s=float(ticks[-1]*.004),
                       max_target_error=error, all_14_channels_all_frames_match=True,
                       tick_delta_histogram={str(k):int(v) for k,v in zip(*np.unique(np.diff(ticks),return_counts=True))},
                       hdf5_sha256=sha(h5), planner_sha256=sha(file), stages=stages,
                       initial_pose_kind='seed_source_reconstruction_before_2500_step_settling_not_actual_snapshot',
                       spawn_pose=spawn(seeds[episode]))
            np.savez(output/f'episode{episode}_derived_timing.npz', physics_tick=ticks,
                     time_s=ticks*.004, recorded_joint_target=observed)
            (output/f'episode{episode}_initial_candidates.json').write_text(json.dumps(row['spawn_pose'],indent=2)+'\n')
            # Save original first-frame camera calibration and JPEG as independently inspectable evidence.
            cameras={}
            for name, group in f['observation'].items():
                cameras[name]={k:np.asarray(group[k][0]).tolist() for k in ('intrinsic_cv','extrinsic_cv','cam2world_gl')}
            (output/f'episode{episode}_cameras.json').write_text(json.dumps(cameras,indent=2)+'\n')
            (output/f'episode{episode}_head_initial.jpg').write_bytes(bytes(f['observation/head_camera/rgb'][0]))
        report['episodes'].append(row)
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
    for e in report['episodes']: print(e['episode'],e['frames'],e['duration_s'],e['max_target_error'],e['tick_delta_histogram'])

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run(a.root,a.cache,a.output)
