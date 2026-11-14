"""Build seed 0 and 2 from the locked seed 0 reset and measured seed 2 reset."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from g2_arx_direct_profile import MODE, PREFIX_CAP
from g2_standalone_deploy import SPEC_FILE, spec
from g2_standalone_package import write_archive
from g2_task_runtime import validate_config
from g2_unified_plan import direct_episode

ROOT = Path(__file__).resolve().parents[3]
GPU = ROOT / "research/g2/arx-stage6-concentrated-20261114/gpu-candidates"
PAIR = ROOT / "research/g2/pair-candidates-20261106"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def seed_two(config):
    source = read(PAIR / "startup-configs/seed2-B-arx-w4.json")
    probe, = [row for row in read(PAIR / "pair-probe.json") if row["episode"] == 2]
    fixture = source["reset_fixture"]
    assert source["scene_seed"] == 2 and fixture["candidate_id"] == "expert-reset-2"
    assert not probe["objects_mismatched"] and probe["geometry"]["ARX-X5"]["clear"]
    assert probe["arx_first_observation"]["ok"]
    assert all(source[key] == config[key] for key in (
        "task", "prompt", "world_task", "episode_info", "calibration",
        "source_world_task", "source_base_world", "arx_base_distance_m", "contact_profile",
    ))
    expected = probe["actual_initial"]["ARX-X5"]
    assert expected["reset_fixture_sha256"] == probe["fixture_sha256"]
    return fixture, expected


def build(label, seed, out):
    parent = GPU / "G1/g2-standalone-arx.tar.gz"
    metadata = read(parent.with_name("package.json"))
    if digest(parent) != metadata["archive_sha256"]:
        raise ValueError("parent archive digest differs")
    output = out / label
    output.mkdir(parents=True)
    with TemporaryDirectory() as temp, tarfile.open(parent) as archive:
        archive.extractall(temp, filter="data")
        release = Path(temp).resolve() / "g2-standalone-arx"
        config_name = lambda episode: f"configs/{episode}.json"
        plan_file = release / "unified-plan.json"
        plan = read(plan_file)
        bindings = plan["trial_bindings"]
        parent_episode = direct_episode(1)
        episode = direct_episode(seed)
        original, = [row for row in plan["trials"] if row["episode_id"] == parent_episode]
        seed_one_file = release / config_name(parent_episode)
        seed_one = read(seed_one_file)

        # Restore the unselected seed 1 roster row for G3. G4 replaces that
        # slot with its own real seed 2 row and cannot dispatch seed 1.
        affected = {config_name(episode), config_name(parent_episode)}
        if seed == 0:
            zero = read(release / config_name(episode))
            fixture, expected = zero["reset_fixture"], bindings[episode]["expected_initial"]
            legacy = dict(seed_one, window=4, max_actions=300,
                          free_space_approach_repair=zero["free_space_approach_repair"])
            legacy.pop("arx_direct")
            write(seed_one_file, legacy)
            bindings[parent_episode]["config_sha256"] = digest(seed_one_file)
            original.update(method="bounded_recovery_and_router", window=4,
                            max_policy_actions=300, recovery_cap=4)
            original.pop("arx_direct")
            trial, = [row for row in plan["trials"] if row["episode_id"] == episode]
        else:
            fixture, expected = seed_two(seed_one)
            trial = original
            seed_one_file.unlink()
            del bindings[parent_episode]

        config = dict(seed_one, reset_fixture=fixture, scene_seed=seed, episode_id=episode)
        validate_config(config)
        fixture_digest = sha256(json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if fixture_digest != expected["reset_fixture_sha256"]:
            raise ValueError("reset geometry does not match the locked evidence")
        config_file = release / config_name(episode)
        write(config_file, config)
        trial.update(seed=seed, episode_id=episode,
                     pair_id=fixture["formal_pair_id"], arx_direct=MODE, method=MODE,
                     window=PREFIX_CAP, max_policy_actions=800, recovery_cap=0,
                     reset_fixture_sha256=fixture_digest, policy_rng_seed=config["policy_rng_seed"])
        bindings[episode] = dict(config_sha256=digest(config_file), expected_initial=expected)
        write(plan_file, plan)
        write(release / SPEC_FILE, spec("arx", seed))
        # Current validator and standalone dispatcher implement the explicit
        # alternate scene binding. The archived simulation and model remain byte-identical.
        affected.update(("unified-plan.json", SPEC_FILE, "scripts/g2_unified_plan.py",
                         "scripts/g2_standalone_deploy.py"))
        for name in ("scripts/g2_unified_plan.py", "scripts/g2_standalone_deploy.py"):
            (release / name).write_bytes((ROOT / name).read_bytes())
        base_file = release / "standalone-base-members.json"
        base = read(base_file)
        for name in affected:
            base["members"].pop(name, None)
        write(base_file, base)
        archive_file = output / parent.name
        write_archive(release, archive_file, "arx")
        write(output / "package.json", dict(
            mode=MODE, scene_seed=seed, policy_rng_seed=config["policy_rng_seed"],
            checkpoint_revision=trial["checkpoint_revision"],
            archive_sha256=digest(archive_file), manifest_sha256=digest(release / "SHA256SUMS"),
            configuration_sha256=digest(release / SPEC_FILE), trial_config_sha256=digest(config_file),
            parent_archive_sha256=metadata["archive_sha256"], reset_fixture_sha256=fixture_digest,
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for label, seed in (("G3", 0), ("G4", 2)):
        build(label, seed, args.output)
