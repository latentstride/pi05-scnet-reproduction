"""URDF-only FK vs recorded world endpose; no simulator or policy weights.

Default-base checks are independent of the trajectory. A separate first-frame
base registration uses episode0 frame0 only, then evaluates every other frame;
it is a diagnostic, not proof of historical simulator/base configuration.
"""
from pathlib import Path
import json
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/robotwin/fk"
OUT.mkdir(parents=True, exist_ok=True)
POSITION_TOL_M = 1e-3
ANGLE_TOL_DEG = 0.1


def rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def transform(p, r):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t


def quat(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class Chain:
    def __init__(self, path, terminal):
        self.tree = ET.parse(path).getroot()
        self.by_child = {j.find('child').get('link'): j for j in self.tree.findall('joint')}
        self.joints = {j.get('name'): j for j in self.tree.findall('joint')}
        j = self.joints[terminal]
        link = j.find('child').get('link')
        chain = []
        while link in self.by_child:
            j = self.by_child[link]
            chain.append(j)
            link = j.find('parent').get('link')
        self.chain = list(reversed(chain))
        self.root_link = link

    def fk(self, values, mode='child'):
        t = np.eye(4)
        for i, j in enumerate(self.chain):
            o = j.find('origin')
            xyz = np.fromstring(o.get('xyz', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get('rpy', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
            r = rot([0, 0, 1], rpy[2]) @ rot([0, 1, 0], rpy[1]) @ rot([1, 0, 0], rpy[0])
            t = t @ transform(xyz, r)
            if mode == 'pre_terminal' and i == len(self.chain)-1:
                break
            kind = j.get('type')
            if kind != 'fixed':
                assert j.get('name') in values, f"Unspecified active ancestor: {j.get('name')}"
                axis = j.find('axis')
                axis = np.fromstring(axis.get('xyz', '1 0 0'), sep=' ') if axis is not None else np.array([1,0,0])
                q = values[j.get('name')]
                if kind in ('revolute', 'continuous'):
                    t = t @ transform([0,0,0], rot(axis, q))
                elif kind == 'prismatic':
                    t = t @ transform(axis*q, np.eye(3))
                else:
                    raise ValueError(kind)
        return t


def errors(pred, target):
    p = np.linalg.norm(pred[:, :3, 3] - target[:, :3, 3], axis=1)
    relative = np.einsum('nji,njk->nik', pred[:, :3, :3], target[:, :3, :3])
    a = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
    return {"frames": len(p), "position_rmse_mm": float(np.sqrt(np.mean(p*p))*1000),
            "position_p95_mm": float(np.percentile(p,95)*1000), "position_max_mm": float(p.max()*1000),
            "angle_rmse_deg": float(np.sqrt(np.mean(a*a))), "angle_p95_deg": float(np.percentile(a,95)),
            "angle_max_deg": float(a.max()), "pass": bool(p.max() <= POSITION_TOL_M and a.max() <= ANGLE_TOL_DEG)}


def main():
    # Independent analytic case: z hinge followed by a fixed 1 m x offset.
    import tempfile
    xml = '<robot name="analytic"><link name="b"/><link name="a"/><link name="e"/><joint name="q" type="revolute"><parent link="b"/><child link="a"/><axis xyz="0 0 1"/></joint><joint name="end" type="fixed"><parent link="a"/><child link="e"/><origin xyz="1 0 0"/></joint></robot>'
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'analytic.urdf'
        path.write_text(xml)
        check = Chain(path, 'end').fk({'q':np.pi/2})
        assert np.allclose(check[:3,3], [0,1,0], atol=1e-12)
        assert np.allclose(check[:3,:3], quat([np.sqrt(.5),0,0,np.sqrt(.5)]), atol=1e-12)
    audit = json.loads((ROOT/'research/robotwin/embodiment_config_audit.json').read_text())
    rows = []
    for cfg in audit['robots']:
        robot = cfg['robot'].lower()
        urdf = ROOT/'research/robotwin/embodiment_assets'/cfg['urdf_member']
        assert hashlib.sha256(urdf.read_bytes()).hexdigest() == cfg['urdf_sha256']
        episodes = sorted((ROOT/'research/robotwin/data_probe'/robot).glob('*.hdf5'))
        for side, arm in enumerate(['left', 'right']):
            chain = Chain(urdf, cfg['ee_joints'][side])
            correction = transform([cfg['gripper_bias']-0.12,0,0], np.eye(3))
            orientation = transform([0,0,0], np.array(cfg['global_trans_matrix']) @ np.array(cfg['delta_matrix']))
            qrows, targets, labels = [], [], []
            for ep in episodes:
                with h5py.File(ep, 'r') as f:
                    q = f[f'joint_action/{arm}_arm'][:]
                    ee = f[f'endpose/{arm}_endpose'][:]
                qrows.extend(q)
                targets.extend(transform(x[:3],quat(x[3:])) for x in ee)
                labels.extend([ep.stem]*len(q))
            targets = np.array(targets)
            pose = cfg['robot_pose_wxyz'][0 if len(cfg['robot_pose_wxyz']) == 1 else side]
            default_base = transform(pose[:3], quat(pose[3:]))
            for mode in ['child', 'pre_terminal']:
                local = np.array([chain.fk(dict(zip(cfg['joint_names'][side], q)), mode) @ orientation @ correction for q in qrows])
                # One rigid transform only, not fitted per episode or per frame.
                registered_base = targets[0] @ np.linalg.inv(local[0])
                per_episode = {}
                for episode in sorted(set(labels)):
                    idx = np.array([i for i,x in enumerate(labels) if x == episode and i != 0])
                    per_episode[episode] = errors(registered_base @ local[idx], targets[idx])
                row = {"robot":robot,"arm":arm,"mode":mode,"root_link":chain.root_link,
                       "joint_chain":[j.get('name') for j in chain.chain],
                       "default_base":default_base.tolist(),
                       "default_base_check":errors(default_base @ local, targets),
                       "episode0_frame0_registered_base":registered_base.tolist(),
                       "registered_held_out_check":errors(registered_base @ local[1:], targets[1:]),
                       "registered_per_episode":per_episode}
                # Diagnostic only: keep configured base orientation, infer one
                # translation and one RIGHT-side tool rotation from a single
                # calibration frame. Never fit separately to held-out episodes.
                diagnostic_base = default_base.copy()
                diagnostic_base[:3,3] += targets[0,:3,3] - (default_base @ local[0])[:3,3]
                tool = np.eye(4)
                tool[:3,:3] = (diagnostic_base @ local[0])[:3,:3].T @ targets[0,:3,:3]
                row['diagnostic_translation_and_tool_calibration'] = {
                    'base':diagnostic_base.tolist(), 'right_tool_rotation':tool[:3,:3].tolist(),
                    'held_out':errors(diagnostic_base @ local[1:] @ tool, targets[1:]),
                    'not_independent_acceptance':True}
                rows.append(row)
    result = {"date":"2026-11-02","position_max_tolerance_mm":POSITION_TOL_M*1000,
              "angle_max_tolerance_deg":ANGLE_TOL_DEG,"thresholds_set_before_run":True,
              "rows":rows,"gpu_used":False,"simulation_run":False,
              "historical_base_pose_independently_verified":False}
    result['analytic_fk_self_check_passed'] = True
    result['gate_passed'] = False
    result['gate_reason'] = 'Actual joint positions and historical joint-global/tool-frame conventions are not independently available; residuals exceed fixed tolerances.'
    (OUT/'fk_results.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(r['robot'],r['arm'],r['mode'],'default',r['default_base_check'],'registered',r['registered_held_out_check'])


if __name__ == '__main__':
    main()
