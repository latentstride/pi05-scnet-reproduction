"""Check the fixed PACE protocol against saved job 127192740 exchanges on CPU."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace
import zipfile
from io import BytesIO

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "research/g2/second-query-audit-127192740"
EPISODE = "unified-seed0-native-w50"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_hash(array):
    array = np.ascontiguousarray(array)
    metadata = json.dumps([array.dtype.str, array.shape], separators=(",", ":"))
    return digest(metadata.encode() + b"\0" + array.tobytes())


def run(protocol, raw):
    lock = read(ROOT / "research/g2/protocol-audit/source-lock.json")
    verified = []
    for entry in lock["files"]:
        source = protocol / entry["file"]
        if source.is_file():
            assert digest(source.read_bytes()) == entry["sha256"], entry["file"]
            verified.append(entry["file"])
    required = {
        "pace/pace/src/pace/rollout.py",
        "pace/description/utils/generate_episode_instructions.py",
        "pace/description/task_instruction/handover_block.json",
        "pace/pace/configs/expert_scenes.json",
        "hf-norm_stats.json",
    }
    assert required.issubset(verified), required - set(verified)
    metrics = read(CASE / "metrics.json")
    submission = read(ROOT / "research/g2/submission-127192740/submission.json")
    package = (
        ROOT
        / "research/g2/final-concentrated-cpu/g2-unified-concentrated-candidate.tar.gz"
    )
    assert digest(package.read_bytes()) == submission["archive_sha256"]
    import tarfile

    with tarfile.open(package) as archive:
        (cfg_member,) = [
            n for n in archive.getnames() if n.endswith(f"/configs/{EPISODE}.json")
        ]
        config_bytes = archive.extractfile(cfg_member).read()
        config = json.loads(config_bytes)
        prefix = cfg_member.split("/configs/")[0]
        model_manifest = json.load(
            archive.extractfile(prefix + "/pace_download_manifest.json")
        )
        runtime_hashes = json.load(archive.extractfile(prefix + "/source-audit.json"))
    assert digest(config_bytes) == metrics["config_sha256"]
    assert model_manifest["norm_stats_sha256"] == digest(
        (protocol / "hf-norm_stats.json").read_bytes()
    )
    with zipfile.ZipFile(CASE / "key-evidence.zip") as evidence:
        events = json.loads(evidence.read("numeric-events.json"))["selected"]
        blocks = [
            np.load(BytesIO(evidence.read(f"action-{i:05}.npy")), allow_pickle=False)
            for i in (1, 2)
        ]
    inputs = [r["event"] for r in events if r["event"].get("kind") == "policy_input"]
    snapshots = [r["event"] for r in events if r["event"].get("kind") == "snapshot"]
    (barrier,) = [
        r["event"]
        for r in events
        if r["event"].get("kind") == "native_approach_barrier"
    ]
    assert [s["stamp"]["tick"] for s in snapshots] == [0, 3575]
    assert (
        barrier["receipt"]["consumed_actions"] == 27
        and barrier["discarded_actions"] == 23
    )
    assert all(i["prompt"] == config["prompt"] for i in inputs)
    assert all(block.shape == (50, 14) for block in blocks)

    cache = read(protocol / "pace/pace/configs/expert_scenes.json")
    shards = cache["tasks"]["handover_block"]
    scene_seeds = [s["scene_seed"] for shard in shards.values() for s in shard]
    assert len(scene_seeds) == len(set(scene_seeds)) == 300
    assert config["scene_seed"] not in scene_seeds
    assert all(s["episode_info"] == {} for shard in shards.values() for s in shard)
    compat = {}
    for name in (
        "envs/handover_block.py",
        "envs/_base_task.py",
        "task_config/demo_clean.yml",
        "task_config/_camera_config.yml",
    ):
        assert (
            digest((protocol / "pace" / name).read_bytes())
            == cache["compatible_source_files_sha256"][name]
        )
        compat[name] = (
            runtime_hashes.get(name) == cache["compatible_source_files_sha256"][name]
            if name in runtime_hashes
            else None
        )

    prompt_spec = read(
        protocol / "pace/description/task_instruction/handover_block.json"
    )
    generator = load_source(
        protocol / "pace/description/utils/generate_episode_instructions.py",
        "pace_instruction_audit",
    )
    generated = []
    for shard in shards.values():
        seed = shard[0]["scene_seed"]
        # Explicit CPU test seeds, not inferred historical batch RNG.
        random.seed(seed)
        rng = np.random.RandomState(seed)
        pool = generator.generate_episode_descriptions(
            "handover_block", [shard[0]["episode_info"]], 100
        )[0]
        generated.append(
            dict(
                scene_seed=seed,
                seen_count=len(set(pool["seen"])),
                unseen_count=len(set(pool["unseen"])),
                example_unseen=str(rng.choice(pool["unseen"])),
                project_prompt_generated=any(
                    config["prompt"] in pool[k] for k in ("seen", "unseen")
                ),
            )
        )
    assert all(config["prompt"] not in prompt_spec[k] for k in ("seen", "unseen"))
    assert not any(row["project_prompt_generated"] for row in generated)

    rollout = load_source(
        protocol / "pace/pace/src/pace/rollout.py", "pace_rollout_audit"
    )
    camera_mapping = dict(
        cam_high="head_camera",
        cam_left_wrist="left_camera",
        cam_right_wrist="right_camera",
    )
    raw_exchange = []
    for i, (entry, block) in enumerate(zip(inputs, blocks, strict=True), 1):
        ep = raw / "trials" / EPISODE / "episode"
        rgb_path = ep / f"observation-{i:05}.npz"
        state = np.asarray(entry["state"], dtype=np.float32)
        with np.load(rgb_path, allow_pickle=False) as rgb:
            native = {
                "joint_action": {"vector": state},
                "observation": {
                    name: {"rgb": rgb[key].transpose(1, 2, 0)}
                    for key, name in camera_mapping.items()
                },
            }
            payload = rollout.encode_observation(native, entry["prompt"])
            hashes = {key: array_hash(payload["images"][key]) for key in camera_mapping}
            assert all(
                hashes[k] == metrics["input"]["camera_stats"][i - 1][k]["array_sha256"]
                for k in hashes
            )
            fields = dict(
                state=array_hash(payload["state"]),
                cameras=hashes,
                prompt=payload["prompt"],
            )
        request_hash = digest(json.dumps(fields, sort_keys=True).encode())
        assert (
            array_hash(block) == metrics["output"]["chunks"][i - 1]["raw_array_sha256"]
        )
        assert (
            digest((ep / f"action-{i:05}.npy").read_bytes())
            == metrics["output"]["chunks"][i - 1]["raw_file_sha256"]
        )
        raw_exchange.append(
            dict(query=i, request_sha256=request_hash, chunk_sha256=array_hash(block))
        )
    assert {k: v for k, v in raw_exchange[0].items() if k != "query"} == metrics[
        "input"
    ]["first_exchange_independently_reproduced"]

    counter = SimpleNamespace(take_action_cnt=0, step_lim=51, eval_success=False)
    queried = []
    counter.get_obs = lambda: native
    counter.get_instruction = lambda: config["prompt"]

    def take_action(_action):
        counter.take_action_cnt += 1

    def infer(_request):
        queried.append(counter.take_action_cnt)
        return {"actions": blocks[0]}

    counter.take_action = take_action
    assert (
        rollout.run_episode(counter, SimpleNamespace(infer=infer), lambda _: 50)[
            "steps"
        ]
        == 51
    )
    assert queried == [0, 50]
    rng_receipt = read(raw / "trials" / EPISODE / "rng-reset.json")
    assert rng_receipt["seed"] == config["policy_rng_seed"] == 20261106
    startup = read(raw / "model/startup.json")
    assert startup["checkpoint_revision"] == lock["revisions"]["checkpoint"]
    assert len(startup["checkpoint_files"]) == len(model_manifest["files"]) == 28
    assert all(
        startup["checkpoint_files"][f["path"]][::2] == [f["bytes"], f["sha256"]]
        for f in model_manifest["files"]
    )
    return dict(
        verified_source_files=verified,
        job_id=submission["job_id"],
        job_runtime_commit=submission["commit"],
        job_archive_sha256=submission["archive_sha256"],
        job_config_sha256=digest(config_bytes),
        revisions=lock["revisions"],
        job_setup={
            k: config[k]
            for k in (
                "scene_seed",
                "policy_rng_seed",
                "window",
                "max_actions",
                "timeout_s",
                "prompt",
            )
        },
        job_fixture=config["reset_fixture"],
        job_observation_ticks=[0, 3575],
        released_scene_shards={
            k: dict(count=len(v), first=v[0], last=v[-1]) for k, v in shards.items()
        },
        project_scene_seed_present_in_cache=False,
        cache_source_file_matches_job_lock=compat,
        prompt_generation_checks=generated,
        saved_exchange_checks=raw_exchange,
        second_request_server_echo=False,
        author_fixed50_control_flow_stub_query_after_actions=queried,
        policy_inference_count=0,
        physics_steps=0,
        gpu_used=False,
        scope="Saved camera encoder parity and source control-flow check. Stub results are not a model rollout.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "raw", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.raw)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
