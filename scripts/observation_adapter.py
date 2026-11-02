"""CPU offline PACE observations; no action relabeling or normalization."""
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'research/robotwin'
CAMERAS = {'cam_high': 'head_camera', 'cam_left_wrist': 'left_camera',
           'cam_right_wrist': 'right_camera'}
DECODER_BLOB = '32ac399ef585a1ee6dea34eb35cb53c2814007b4'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_decoder():
    path = DATA / 'stage3_sources/decode_image_bit.py'
    code = path.read_bytes()
    if hashlib.sha1(f'blob {len(code)}\0'.encode() + code).hexdigest() != DECODER_BLOB:
        raise ValueError('Unaudited decoder')
    spec = importlib.util.spec_from_file_location('stage3_codec', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_image_bit


def check_observation(obs):
    if set(obs) != {'state', 'images', 'prompt'}:
        raise ValueError('Unexpected observation keys')
    state = obs['state']
    if state.shape != (14,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError('state must be finite float32[14]')
    if not ((state[[6, 13]] >= 0) & (state[[6, 13]] <= 1)).all():
        raise ValueError('Grippers outside [0,1]')
    if set(obs['images']) != set(CAMERAS):
        raise ValueError('Three mapped RGB cameras required')
    for image in obs['images'].values():
        if image.shape != (3, 240, 320) or image.dtype != np.uint8:
            raise ValueError('Expected uint8 CHW[3,240,320]')
    if not isinstance(obs['prompt'], str) or not obs['prompt'].strip():
        raise ValueError('Missing instruction')


class EpisodeAdapter:
    """Verified sampled episode. Call observation(frame) or iterate observations().

    provenance is separate from the exact PACE input dictionary. Files remain read-only.
    """
    def __init__(self, robot, episode, root=None, split='seen', instruction_index=0):
        self.root = Path(root) if root else DATA / 'data_probe'
        manifest = json.loads((DATA / 'sample_manifest.json').read_text())
        archive = next(a for a in manifest['archives'] if a['robot'] == robot)
        member = next(m for m in archive['selected'] if Path(m['file']).stem == episode)
        path = self.root / member['file']
        if sha256(path) != member['sha256']:
            raise ValueError('HDF5 SHA-256 mismatch')
        relative = f'{robot}/metadata/{episode}.json'
        meta = next(m for m in archive['metadata_selected'] if m['file'] == relative)
        if sha256(self.root / relative) != meta['sha256']:
            raise ValueError('Instruction SHA-256 mismatch')
        if instruction_index < 0:
            raise ValueError('Instruction index must be nonnegative')
        self.prompt = json.loads((self.root / relative).read_text())[split][instruction_index]
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError('Empty instruction')
        self.provenance = {'robot': robot, 'episode': episode, 'hdf5': member['file'],
                           'hdf5_sha256': member['sha256'], 'dataset_revision': manifest['revision'],
                           'language': {'file': relative, 'sha256': meta['sha256'], 'split': split,
                                        'index': instruction_index, 'text': self.prompt},
                           'state_source': 'recorded_joint_target', 'fps': None, 'timestamps': None,
                           'actual_qpos': None, 'derived_next_state_action': None,
                           'policy_predicted_action': None}
        self.file = h5py.File(path, 'r')
        self.frames = len(self.file['joint_action/vector'])
        self.decode = load_decoder()

    def observation(self, frame):
        if not 0 <= frame < self.frames:
            raise IndexError(frame)
        obs = {'state': self.file['joint_action/vector'][frame].astype(np.float32),
               'images': {dest: np.ascontiguousarray(self.decode(
                   self.file[f'observation/{src}/rgb'][frame]).transpose(2, 0, 1))
                   for dest, src in CAMERAS.items()}, 'prompt': self.prompt}
        check_observation(obs)
        return obs

    def observations(self):
        for frame in range(self.frames):
            yield self.observation(frame)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', required=True, choices=['aloha-agilex', 'arx-x5'])
    parser.add_argument('--episode', default='episode0')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    with EpisodeAdapter(args.robot, args.episode, args.root) as adapter:
        obs = adapter.observation(args.frame)
        print(json.dumps({'provenance': adapter.provenance, 'frame': args.frame,
                          'state': obs['state'].tolist(), 'image_shapes': {
                              k: list(v.shape) for k, v in obs['images'].items()}}, indent=2))
