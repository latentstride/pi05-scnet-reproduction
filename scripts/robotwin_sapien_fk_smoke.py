"""Static CPU-only SAPIEN/URDF correspondence; no renderer, planner or policy.

Original URDFs are hash checked. Only visual/collision/material elements are
removed in separate output copies. Three imposed joint states are read back
from PhysX, without stepping dynamics. Drive targets are deliberately different
from actual qpos to expose their distinct meanings, not measure tracking error.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from verify_robotwin_fk import Chain, errors, quat, rot, transform

ROOT = Path(__file__).resolve().parents[1]
POS_TOL_M = 1e-3
ANGLE_TOL_DEG = 0.1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_matrix(pose):
    return transform(np.asarray(pose.p, dtype=float), quat(np.asarray(pose.q)))


def compare(a, b):
    return errors(np.asarray([a]), np.asarray([b]))


def endpose(joint_world, cfg, tcp=False):
    """RoboTwin robot.py _trans_endpose at pinned 13c3c47, in world axes."""
    out = joint_world.copy()
    out[:3, :3] = joint_world[:3, :3] @ np.asarray(cfg['global_trans_matrix']) @ np.asarray(cfg['delta_matrix'])
    distance = cfg['gripper_bias'] - (0 if tcp else 0.12)
    out[:3, 3] += out[:3, :3] @ np.array([distance, 0, 0])
    return out


def prepare(out):
    audit = json.loads((ROOT/'research/robotwin/embodiment_config_audit.json').read_text())
    assets = ROOT/'research/robotwin/embodiment_assets'
    prepared = []
    for entry in audit['robots']:
        original = assets/entry['urdf_member']
        config_path = assets/entry['archive_config_member']
        assert digest(original) == entry['urdf_sha256'], original
        assert digest(config_path) == entry['config_sha256'], config_path
        cfg = yaml.safe_load(config_path.read_text())
        tree = ET.parse(original)
        removed = {'visual': 0, 'collision': 0, 'material': 0}
        # Inertials, origins, axes, limits and joint/link names are retained.
        for parent in tree.getroot().iter():
            for child in list(parent):
                if child.tag in removed:
                    removed[child.tag] += 1
                    parent.remove(child)
        target = out/(entry['robot'].lower()+'-kinematics.urdf')
        tree.write(target, encoding='utf-8', xml_declaration=True)
        assert [ET.tostring(j) for j in ET.parse(original).getroot().findall('joint')] == [
            ET.tostring(j) for j in tree.getroot().findall('joint')], 'Joint definitions changed'
        for side in range(2):
            chain = Chain(original, cfg['ee_joints'][side])
            assert chain.chain[-1].find('child').get('link') == cfg['move_group'][side]
        prepared.append((entry, cfg, original, target, removed))
    manifest = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'prepared', 'gpu_used': False,
        'renderer_created': False, 'physics_stepped': False,
        'full_assets_downloaded': False,
        'position_tolerance_m': POS_TOL_M, 'angle_tolerance_deg': ANGLE_TOL_DEG,
        'robots': [{
            'robot': e['robot'], 'urdf_sha256': digest(original),
            'config_sha256': e['config_sha256'],
            'kinematics_urdf': target.name, 'kinematics_urdf_sha256': digest(target),
            'removed_geometry_elements': removed,
            'urdf_joint_elements_unchanged': True,
        } for e, cfg, original, target, removed in prepared],
    }
    (out/'prepared_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return prepared, manifest


def run(out, prepared, manifest, arx_distance):
    import sapien
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    records = []
    for entry, cfg, original, target, removed in prepared:
        entities = []
        instance_count = 1 if cfg['dual_arm'] else 2
        for index in range(instance_count):
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            entity = loader.load(str(target))
            assert entity is not None
            base = np.array(cfg['robot_pose'][0], dtype=float)
            base[3:] /= np.linalg.norm(base[3:])
            if not cfg['dual_arm']:
                base[0] += (-0.5 if index == 0 else 0.5)*arx_distance
            entity.set_root_pose(sapien.Pose(base[:3], base[3:]))
            entities.append(entity)
        xml = ET.parse(original).getroot()
        by_child = {j.find('child').get('link'): j for j in xml.findall('joint')}
        arm_names = set(sum(cfg['arm_joints_name'], []))
        for case, sign in [('neutral', 0), ('offset_positive', 1), ('offset_negative', -1)]:
            for instance, entity in enumerate(entities):
                active = entity.get_active_joints()
                assert all(j.get_dof() == 1 for j in active), 'Only scalar URDF joints supported'
                names = [j.get_name() for j in active]
                assert len(names) == entity.get_dof() and len(set(names)) == len(names)
                limits = np.asarray(entity.get_qlimits(), dtype=float)
                q = np.clip(np.zeros(len(names)), limits[:, 0], limits[:, 1])
                for i, name in enumerate(names):
                    if name in arm_names:
                        q[i] = sign*(0.08 + 0.015*(i % 6))*(-1 if i % 2 else 1)
                    elif active[i].get_type() == 'prismatic':
                        q[i] = (limits[i, 0]+limits[i, 1])/2
                q = np.clip(q, limits[:, 0], limits[:, 1])
                entity.set_qpos(q.astype(np.float32))
                entity.set_qvel(np.zeros_like(q, dtype=np.float32))
                actual = np.asarray(entity.get_qpos(), dtype=float).copy()
                for i, joint in enumerate(active):
                    joint.set_drive_properties(float(cfg['joint_stiffness']), float(cfg['joint_damping']))
                    goal = actual[i]+(0.025 if names[i] in arm_names else 0)
                    goal = np.clip(goal, limits[i, 0], limits[i, 1])
                    joint.set_drive_target(np.array([goal], dtype=np.float32))
                np.testing.assert_allclose(entity.get_qpos(), actual, rtol=0, atol=1e-7)
                goals = np.array([np.asarray(j.get_drive_target()).item() for j in active])
                qmap = dict(zip(names, actual))
                base_world = pose_matrix(entity.get_root_pose())
                joint_records = []
                for joint in entity.get_joints():
                    parent, child = joint.get_parent_link(), joint.get_child_link()
                    if parent is None:
                        continue
                    parent_world, child_world = pose_matrix(parent.entity.pose), pose_matrix(child.entity.pose)
                    in_parent, in_child = pose_matrix(joint.get_pose_in_parent()), pose_matrix(joint.get_pose_in_child())
                    actual_joint_world = pose_matrix(joint.get_global_pose())
                    motion = np.eye(4)
                    value = qmap.get(joint.get_name(), 0.)
                    if joint.get_type() in ('revolute', 'revolute_unwrapped'):
                        motion[:3, :3] = rot([1, 0, 0], value)
                    elif joint.get_type() == 'prismatic':
                        motion[0, 3] = value
                    joint_records.append({
                        'name': joint.get_name(), 'type': joint.get_type(),
                        'parent_link': parent.get_name(), 'child_link': child.get_name(),
                        'pose_in_parent': in_parent.tolist(), 'pose_in_child': in_child.tolist(),
                        'parent_link_world': parent_world.tolist(), 'child_link_world': child_world.tolist(),
                        'joint_global_world': actual_joint_world.tolist(),
                        'axis_frame_closure': compare(parent_world @ in_parent @ motion, child_world @ in_child),
                        'global_vs_parent_frame': compare(actual_joint_world, parent_world @ in_parent),
                        'global_vs_child_frame': compare(actual_joint_world, child_world @ in_child),
                    })
                arms = []
                sides = range(2) if cfg['dual_arm'] else [instance]
                for side in sides:
                    joint = next(j for j in entity.get_joints() if j.get_name() == cfg['ee_joints'][side])
                    chain = Chain(original, cfg['ee_joints'][side])
                    child_fk = base_world @ chain.fk(qmap)
                    parent_name = joint.get_parent_link().get_name()
                    parent_fk = base_world if parent_name == chain.root_link else base_world @ Chain(
                        original, by_child[parent_name].get('name')).fk(qmap)
                    child_candidate = child_fk @ pose_matrix(joint.get_pose_in_child())
                    parent_candidate = parent_fk @ pose_matrix(joint.get_pose_in_parent())
                    measured_joint_world = pose_matrix(joint.get_global_pose())
                    measured_end = endpose(measured_joint_world, cfg)
                    arms.append({
                        'side': ['left', 'right'][side], 'ee_joint': joint.get_name(),
                        'ee_child_link': joint.get_child_link().get_name(),
                        'urdf_child_fk_world': child_fk.tolist(),
                        'sapien_child_link_world': pose_matrix(joint.get_child_link().entity.pose).tolist(),
                        'child_fk_check': compare(child_fk, pose_matrix(joint.get_child_link().entity.pose)),
                        'robotwin_ee_world': measured_end.tolist(),
                        'robotwin_tcp_world': endpose(measured_joint_world, cfg, tcp=True).tolist(),
                        'ee_from_urdf_child_joint_frame': compare(endpose(child_candidate, cfg), measured_end),
                        'ee_from_urdf_parent_joint_frame': compare(endpose(parent_candidate, cfg), measured_end),
                    })
                records.append({
                    'robot': entry['robot'], 'instance': instance, 'case': case,
                    'physical_step': 0, 'joint_order': names,
                    'commanded_qpos': q.tolist(), 'actual_qpos': actual.tolist(),
                    'drive_targets': goals.tolist(),
                    'drive_target_qpos_distinct': bool(np.max(np.abs(goals-actual)) > 0.01),
                    'root_pose_world': base_world.tolist(),
                    'joints': joint_records, 'arms': arms,
                })
        assert digest(original) == entry['urdf_sha256'], 'Original URDF changed'
    arms = [a for row in records for a in row['arms']]
    joints = [j for row in records for j in row['joints']]
    candidates = [name for name in ['child', 'parent'] if all(
        a['ee_from_urdf_'+name+'_joint_frame']['pass'] for a in arms)]
    summary = {
        'snapshots': len(records), 'arm_state_checks': len(arms), 'joint_frame_checks': len(joints),
        'child_fk_passed': all(a['child_fk_check']['pass'] for a in arms),
        'joint_axis_closure_passed': all(j['axis_frame_closure']['pass'] for j in joints),
        'consistent_ee_joint_frame_conventions': candidates,
        'drive_target_qpos_separation_observed': all(row['drive_target_qpos_distinct'] for row in records),
        'child_fk_position_max_mm': max(a['child_fk_check']['position_max_mm'] for a in arms),
        'child_fk_angle_max_deg': max(a['child_fk_check']['angle_max_deg'] for a in arms),
    }
    passed = (summary['child_fk_passed'] and summary['joint_axis_closure_passed'] and bool(candidates)
              and summary['drive_target_qpos_separation_observed'])
    result = {**manifest, 'status': 'passed' if passed else 'failed',
              'sapien_version': importlib.metadata.version('sapien'),
              'arx_base_distance_m': arx_distance,
              'base_pose_scope': 'Explicit diagnostic fixture, not recovered historical collection poses',
              'historical_hdf5_correspondence_validated': False,
              'rendering_validated': False, 'planning_validated': False,
              'summary': summary, 'snapshots': records}
    (out/'sapien_fk_smoke.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], **summary}, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'research/robotwin/sapien-fk-smoke')
    parser.add_argument('--prepare-only', action='store_true', help='Validate and strip URDF geometry without importing SAPIEN')
    parser.add_argument('--arx-base-distance', type=float, default=0.6, help='Explicit diagnostic left/right base separation in metres')
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT.resolve())
    assert np.isfinite(args.arx_base_distance) and args.arx_base_distance >= 0
    out.mkdir(parents=True, exist_ok=True)
    prepared, manifest = prepare(out)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return 0
    return run(out, prepared, manifest, args.arx_base_distance)


if __name__ == '__main__':
    raise SystemExit(main())
