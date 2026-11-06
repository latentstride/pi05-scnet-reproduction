"""Run a roster-bound G2 episode through the frozen diagnostic control path."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np

from g2_adapter import images_of, read_snapshot
from g2_collision import JointCollisionWorld
from g2_episode import save_array, save_json
from g2_export import episode_record
from g2_initial import compare_initial, digest_json
from g2_orchestrator import interrupt_signals
from g2_run_task import execute
from g2_task_runtime import inspect_assets, real_scene, validate_config, verify_runtime
from pace_runtime import open_policy
from summarize_experiments import LOCK_FIELDS, SCHEMA, decode, validate_episode

SIDES = ('left', 'right')
ROBOTS = {'aloha-agilex': 'aloha-agilex', 'arx-x5': 'ARX-X5'}


def load_trial(lock_path, expected_sha, episode_id, config_path):
    raw = lock_path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha:
        raise ValueError('run lock hash mismatch')
    lock = decode(raw.decode())
    if lock.get('schema_version') != 1 or lock.get('status') != 'frozen':
        raise ValueError('frozen run lock required')
    roster = lock['episodes']
    if any(set(row) != set(LOCK_FIELDS) for row in roster):
        raise ValueError('invalid scoring roster fields')
    rows = {row['episode_id']: row for row in roster}
    bindings = lock['bindings']
    if len(rows) != len(roster) or set(rows) != set(bindings):
        raise ValueError('run lock must bind each roster entry exactly once')
    row, binding = rows[episode_id], bindings[episode_id]
    if row['pair_id'] is None or row['comparison_id'] is None:
        raise ValueError('locked policy trials require a pair and comparison')
    config_bytes = config_path.read_bytes()
    if sha256(config_bytes).hexdigest() != binding['config_sha256']:
        raise ValueError('per-episode configuration hash mismatch')
    config = decode(config_bytes.decode())
    validate_config(config)
    condition = row['condition']
    robot = ROBOTS[condition['embodiment']]
    source = 'source_policy' if robot == 'aloha-agilex' else 'target_policy'
    if (row['seed'], row['task'], condition['id'], condition['window_actions'], robot, source) != (
            config['scene_seed'], config['task'], config['condition'], config['window'],
            config['embodiment'], condition['action_source']):
        raise ValueError('episode configuration does not match the scoring roster')
    fixture = config['reset_fixture']
    expected = binding['expected_initial']
    if (fixture['formal_pair_id'] != row['pair_id']
            or digest_json(fixture) != expected['reset_fixture_sha256']
            or abs(fixture['block_mass_kg']-.01)>1e-8):
        raise ValueError('locked pair identity, reset hash or mass mismatch')
    config['episode_id'] = episode_id
    identity = dict(row, run_lock_sha256=expected_sha,
                    trajectory_reference_id=binding['trajectory_reference_id'])
    return config, expected, identity


class InitialMismatch(ValueError):
    """Measured scene fails the locked initial-state comparison."""


class VerifiedReset:
    """Verify before the first policy query, then delegate to the frozen backend."""

    def __init__(self, backend, expected, output, verify_scene):
        self.backend = backend
        self.expected = expected
        self.output = output
        self.verify_scene = verify_scene
        self.status = 'unverified'

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def reset(self, spec):
        actual = self.backend.reset(spec)
        save_json(self.output/'initial-readback.json', actual)
        mismatches = compare_initial(actual, self.expected)
        if actual['reset_fixture_sha256'] != self.expected['reset_fixture_sha256']:
            mismatches.append('reset_fixture_sha256')
        if mismatches:
            self.status = 'unmatched'
            raise InitialMismatch(', '.join(mismatches))
        snapshot, _ = self.backend.capture()
        rgb = images_of(snapshot)
        checks = self.verify_scene(self.backend, snapshot, rgb)
        save_array(self.output/'initial-rgb.npz', np.savez_compressed, **rgb)
        save_json(self.output/'initial-verification.json', dict(checks, matched=True))
        self.status = 'matched'
        return dict(actual, common_workspace_pass=True,
                    workspace_check_id='locked_initial_real_scene_fcl_v1')


def verify_real_scene(backend, snapshot, rgb):
    import sapien

    env = backend.env
    block = getattr(env.box, 'actor', env.box)
    body = block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    mass = float(body.mass) if body is not None else float('nan')
    if not np.isfinite(mass) or abs(mass-.01)>1e-8:
        raise InitialMismatch('actual block mass differs from locked 10 g task')
    for name, pixels in rgb.items():
        if np.min(pixels) == np.max(pixels):
            raise ValueError(f'{name} initial rendered image is constant')
    q, fingers, _, _ = read_snapshot(snapshot, backend.arms)
    if hasattr(backend, 'collision'):
        collision = backend.collision
        collision.refresh()
        clear = getattr(collision, 'measured_clear', collision)(q, fingers)
    else:
        entities = [env.robot.left_entity, env.robot.right_entity]
        collision = JointCollisionWorld.from_scene(backend.actual_scene, entities,
            [arm.chain.names for arm in backend.arms], [arm.grip.names for arm in backend.arms])
        clear = collision(q, fingers)
    if not clear:
        raise InitialMismatch(f'actual initial geometry blocked: {collision.last_collisions}')
    return dict(block_mass_kg=mass, fcl_clear=True, image_shapes={k:list(v.shape) for k,v in rgb.items()})


def verify_model_receipt(checkpoint, manifest_path, startup_path):
    raw = manifest_path.read_bytes()
    manifest = decode(raw.decode())
    startup = decode(startup_path.read_text())
    files = startup.get('checkpoint_files', {})
    expected = {entry['path']:(entry['bytes'], entry['sha256']) for entry in manifest['files']}
    actual = {path:(record[0], record[2]) for path,record in files.items()}
    if (checkpoint['manifest_sha256'] != sha256(raw).hexdigest()
            or checkpoint['revision'] != manifest['revision']
            or checkpoint['norm_stats_sha256'] != manifest['norm_stats_sha256']
            or startup.get('status') != 'ready'
            or startup.get('checkpoint_revision') != manifest['revision']
            or actual != expected):
        raise ValueError('model startup receipt differs from the locked checkpoint')


def run_checked(config, expected, identity, factory, policy, output, *, verify_scene):
    """Run the measured reset gate and retain every terminal diagnostic record."""
    output = Path(output).resolve()
    reset_gate = None

    @contextmanager
    def checked_factory(spec):
        nonlocal reset_gate
        with factory(spec) as (backend, adapter):
            reset_gate = VerifiedReset(backend, expected, output, verify_scene)
            yield reset_gate, adapter

    result = execute(config, checked_factory, policy, output)
    status = reset_gate.status if reset_gate is not None else 'unverified'
    mismatch = result['failure'] and result['failure']['code'] == 'InitialMismatch'
    if mismatch:
        status = 'unmatched'
    row = episode_record(output, dict(identity, match_status=status))
    if mismatch:
        row['initial'] = decode((output/'initial-readback.json').read_text())
        row.update(outcome='initial_state_mismatch')
        row['failure']['stage'] = 'match'
    elif result['outcome'] == 'timeout':
        # A wall-clock ceiling is an infrastructure outcome, not a task failure.
        row.update(outcome='interrupted')
        row['failure']['stage'] = 'infrastructure'
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('lock','config','pace-root','asset-audit','source-audit','model-manifest','model-startup','output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('lock-sha','episode-id','uri'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--policy-timeout-s', type=float, default=120.)
    p.add_argument('--startup-timeout-s', type=float,
                   help='run at most four actions and export startup evidence without a score')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_NAME') != 'cpf-pace-g2':
        raise RuntimeError('locked execution requires the existing G2 allocation')
    config, expected, identity = load_trial(a.lock, a.lock_sha, a.episode_id, a.config)
    lock_bytes = a.lock.read_bytes()
    if sha256(lock_bytes).hexdigest() != a.lock_sha:
        raise ValueError('run lock changed during setup')
    lock = decode(lock_bytes.decode())
    for kind, path in (('asset_audit', a.asset_audit), ('source_audit', a.source_audit)):
        if sha256(path.read_bytes()).hexdigest() != lock['artifacts'][kind]:
            raise ValueError(f'{kind} hash differs from the run lock')
    verify_model_receipt(identity['condition']['checkpoint'], a.model_manifest, a.model_startup)
    output = a.output.resolve()
    pace = a.pace_root.resolve()
    source = decode(a.source_audit.read_text())
    verify_runtime(pace, source)
    inspect_assets(config, pace/'assets', a.asset_audit)
    if a.policy_timeout_s <= 0 or not np.isfinite(a.policy_timeout_s):
        raise ValueError('invalid policy timeout')
    if a.startup_timeout_s is not None:
        if not np.isfinite(a.startup_timeout_s) or a.startup_timeout_s <= 0:
            raise ValueError('invalid startup timeout')
        config = dict(config, max_actions=min(4,config['max_actions']), timeout_s=a.startup_timeout_s)

    def factory(spec):
        return real_scene(config, spec, pace, pace/'assets', a.asset_audit.resolve(), source, output)

    with interrupt_signals(), open_policy(a.uri, 'checkpoint', a.policy_timeout_s) as policy:
        row = run_checked(config, expected, identity, factory, policy, output, verify_scene=verify_real_scene)
    if a.startup_timeout_s is not None:
        result = decode((output/'result.json').read_text())
        passed = (row['match_status'] == 'matched' and result['executed'] > 0 and
                  result['outcome'] in ('success','task_unsuccessful'))
        save_json(output/'startup-check.json', dict(passed=passed, evaluation=row, measured_runtime=result))
        if not passed:
            raise SystemExit(1)
        return
    row['data_kind'] = 'formal'
    Draft202012Validator(decode(SCHEMA.read_text())).validate(row)
    validate_episode(row)
    save_json(output/'formal-record.json', row)


if __name__ == '__main__':
    main()
