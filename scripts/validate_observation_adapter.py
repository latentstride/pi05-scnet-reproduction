"""Six-episode CPU schema gate, reusing archived complete image decode evidence."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import h5py
import numpy as np
from observation_adapter import DATA, EpisodeAdapter, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(root=None):
    manifest = json.loads((DATA / 'sample_manifest.json').read_text())
    previous = json.loads((DATA / 'probe_results.json').read_text())
    rows = []
    for archive in manifest['archives']:
        robot = archive['robot']
        audit = json.loads((DATA / 'embodiment_config_audit.json').read_text())
        require(audit['revision'] == manifest['revision'], 'Embodiment config revision')
        config = next(c for c in audit['robots'] if c['robot'].lower() == robot)
        config_text = config['config_text']
        names = ([['fl_joint'+str(i) for i in range(1, 7)],
                  ['fr_joint'+str(i) for i in range(1, 7)]] if robot == 'aloha-agilex'
                 else [['joint'+str(i) for i in range(1, 7)]] * 2)
        # Parse only the audited, YAML flow-style list with Python's literal parser.
        import ast
        config_names = ast.literal_eval(re.search(r'^arm_joints_name:\s*(.*?)(?=^\w|\Z)', config_text, re.M | re.S)[1].strip())
        require(config_names == names, 'Config joint order')
        config_hash = hashlib.sha256(config_text.encode()).hexdigest()
        require(config_hash == config['config_sha256'], 'Config SHA-256')
        for member in archive['selected']:
            episode = Path(member['file']).stem
            with EpisodeAdapter(robot, episode, root) as adapter:
                f = adapter.file
                t = adapter.frames
                fields = {}
                f.visititems(lambda k,v: fields.update({k:v}) if isinstance(v,h5py.Dataset) else None)
                require(all(v.shape and v.shape[0] == t for v in fields.values()), 'Field lengths')
                require(all(np.isfinite(v[:]).all() for v in fields.values() if v.dtype.kind in 'fiu'), 'Nonfinite raw field')
                require(f['joint_action/vector'].shape == (t,14), '14D vector')
                require(not any(re.search(r'(^|/)(timestamps?|fps|dt|time)$',k,re.I) for k in fields), 'New timing field needs audit')
                joined = []
                quat_errors = {}
                for arm in ['left','right']:
                    require(f[f'joint_action/{arm}_arm'].shape == (t,6), 'Arm dimensions')
                    require(f[f'endpose/{arm}_endpose'].shape == (t,7), 'Endpose dimensions')
                    gripper = f[f'joint_action/{arm}_gripper'][:]
                    require(gripper.shape == (t,), 'Gripper dimensions')
                    require(((gripper >= 0)&(gripper <= 1)).all(), 'Gripper range')
                    require(np.array_equal(gripper,f[f'endpose/{arm}_gripper'][:]), 'Duplicate gripper mismatch')
                    joined.extend([f[f'joint_action/{arm}_arm'][:],gripper[:,None]])
                    err = float(np.abs(np.linalg.norm(f[f'endpose/{arm}_endpose'][:,3:],axis=1)-1).max())
                    require(err < 1e-5, 'Quaternion norm')
                    quat_errors[arm] = err
                require(np.array_equal(np.concatenate(joined,axis=1),f['joint_action/vector'][:]), 'Left/right vector order')
                # Check every state; expensive image evidence reused only after matching file hashes.
                states = f['joint_action/vector'][:].astype(np.float32)
                require(states.shape == (t,14) and np.isfinite(states).all(), 'All float32 states')
                old = next(e for e in previous['episodes'] if e['robot']==robot and e['episode']==episode)
                require(old['sha256']==member['sha256'] and previous['decoder_git_blob']=='32ac399ef585a1ee6dea34eb35cb53c2814007b4', 'Prior evidence identity')
                require(all(i['frames_decoded']==t and i['decoded_shapes_hwc']==[[240,320,3]] and i['dtype']=='uint8' for i in old['images'].values()), 'Archived decode schema')
                frames = sorted({0,t//2,t-1})
                for frame in frames:
                    obs = adapter.observation(frame)
                    require(np.array_equal(obs['state'], states[frame]), 'Adapter changed state')
                rows.append({**adapter.provenance,'frames':t,'all_schema_assertions_passed':True,
                             'all_state_frames_checked':t,'fresh_adapter_frames':frames,
                             'fresh_rgb_decodes':len(frames)*3,'joint_names':names,
                             'config_sha256':config_hash,'quaternion_max_norm_error':quat_errors,
                             'reused_rgb_decodes':sum(i['frames_decoded'] for i in old['images'].values())})
                print('PASS',robot,episode,t,flush=True)
    report = {'schema_version':1,'all_schema_assertions_passed':len(rows)==6,
              'episodes':rows,'total_state_frames':sum(r['frames'] for r in rows),
              'fresh_rgb_decodes':sum(r['fresh_rgb_decodes'] for r in rows),
              'reused_rgb_decodes':sum(r['reused_rgb_decodes'] for r in rows),
              'probe_evidence_sha256':sha256(DATA/'probe_results.json'),
              'decoder_sha256':sha256(DATA/'stage3_sources/decode_image_bit.py'),
              'fps':None,'simulation_replay_passed':False,'model_inference_tested':False,'gpu_used':False}
    (DATA/'observation_adapter_validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    main(parser.parse_args().root)
