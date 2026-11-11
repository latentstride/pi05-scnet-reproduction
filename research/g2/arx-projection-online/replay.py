"""CPU PhysX and wire-protocol test. Camera bytes and server responses are test fixtures."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import numpy as np
from websockets.sync.server import serve

from arx_replay import EPISODE, build_adapter, digest, load
from g2_arx_transfer_probe_cpu import restart_backend, save
from g2_final_trial import RuntimeReset, classify
from g2_locked_task import verify_real_scene
from g2_remaining_conditions import run_condition
from g2_remaining_coordinator import RecoveryCoordinator
from g2_task_tcp_projection import METHOD
from g2_task_runtime import validate_config
from pace_runtime import TASK, decode, msgpack_numpy, open_policy, validate_request


def replay(query, args, config, row, initial, reference, chunk):
    output = args.output / str(query)
    output.mkdir()
    def build(name):
        directory = output / name
        directory.mkdir()
        return build_adapter(config, args.assets, args.release / "asset-audit.json",
                             initial, reference["measured"], reference["request"], directory)

    _, old_adapter, snapshot, drive = build("original")
    old = RecoveryCoordinator(old_adapter, SimpleNamespace(_drive_fingers=lambda: drive),
                              approach_repair=True)
    from g2_adapter import G2Error
    try:
        old_plan, _ = old.action_plan(chunk, snapshot, 4, lambda: 90., lambda _: None)
        old_count = len(old_plan.waypoint_ticks)
    except G2Error:
        old_count = 0
    assert old_count == {35: 2, 36: 0}[query]

    scene, adapter, snapshot, drive = build("online")
    backend, _, _ = restart_backend(scene, adapter, snapshot, drive, config, initial, args.assets)
    adapter.last_stamp = None
    actors = {actor.name: actor for actor in scene.get_all_actors()}
    backend.env.target_box = actors["target_box"]
    backend.env.table = actors["table"]
    lifecycle = []
    backend.env.close_env = lambda **_: lifecycle.append("closed")
    backend.env.check_success = lambda: False
    backend.reset = lambda _: initial
    backend.video = SimpleNamespace(frame=lambda _: None, close=lambda: None, path=None)
    names = ("head_camera", "left_camera", "right_camera")
    pattern = np.arange(192, dtype=np.uint8).reshape(8, 8, 3)
    backend.env.get_obs = lambda: {"observation": {
        name: {"rgb": pattern + ((backend.tick + index) % 256)}
        for index, name in enumerate(names)
    }}
    capture = backend.capture

    def measured_capture():
        adapter.source_collision.refresh()
        adapter.target_collision.refresh()
        return capture()

    backend.capture = measured_capture
    requests = []
    replies = []

    def handler(socket):
        socket.send(msgpack_numpy.packb(dict(backend="checkpoint", task=TASK, protocol=1)))
        for index in range(2):
            request = decode(socket.recv(timeout=240))
            validate_request(request)
            requests.append(request)
            if index == 0:
                actions = chunk
            else:
                # Test-only response derived from the newly received state.
                actions = np.tile(request["state"], (50, 1))
                if query == 36:
                    actions[0, 6] = 2
            replies.append(actions)
            socket.send(msgpack_numpy.packb({"actions": actions}))

    cleanup_failures = []
    runtime = RuntimeReset(backend, initial, output / "episode", verify_real_scene, cleanup_failures)

    @contextmanager
    def factory(_):
        yield runtime, adapter
        runtime.cleanup["scene_context_closed"] = True
        if runtime.candidate_stop:
            runtime.candidate_stop["scene_context_closed"] = True

    with serve(handler, "127.0.0.1", 0, compression=None) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with open_policy(f"ws://127.0.0.1:{server.socket.getsockname()[1]}", "checkpoint", 240) as policy:
                result = run_condition(dict(config, max_actions=8), row, factory, policy, output / "episode")
        finally:
            server.shutdown()
            thread.join()
    import json
    events = [json.loads(line) for line in (output / "episode/events.jsonl").open()]
    sent = [event for event in events if event["kind"] == "policy_request"]
    corrections = [event for event in events if event["kind"] == "task_tcp_projection"]
    discarded = [event for event in events if event["kind"] == "projection_chunk_discarded"]
    approved = [event for event in events if event["kind"] == "approved_plan"]
    assert len(corrections) == len(discarded) == 1 and discarded[0]["discarded_rows"] == 46, result
    assert len(sent) == len(requests) == 2 and sent[1]["after_projection_from"] == sent[0]["stamp"], result
    assert sent[1]["stamp"]["tick"] > sent[0]["stamp"]["tick"]
    assert not np.array_equal(requests[0]["state"], requests[1]["state"])
    assert not np.array_equal(replies[0], replies[1])
    for index, event in enumerate(sent, 1):
        assert {"block", "target", "table"} <= event["objects_task"].keys()
        for arm in event["actual"]:
            assert arm["stamp"] == event["stamp"] and len(arm["qpos"]) == len(arm["names"])
        with np.load(output / "episode" / f"observation-{index:05}.npz") as captured:
            assert all(np.array_equal(captured[key], image) for key, image in requests[index - 1]["images"].items())
        with (output / "episode" / f"action-{index:05}.npy").open("rb") as stream:
            assert np.array_equal(np.load(stream, allow_pickle=False), replies[index - 1])
    assert all(not np.array_equal(requests[0]["images"][key], requests[1]["images"][key])
               for key in requests[0]["images"])
    category = classify(result, candidate_stop=runtime.candidate_stop)
    assert not cleanup_failures and not result.get("cleanup_failures") and lifecycle == ["closed"], result
    assert runtime.cleanup["hold_complete"] and runtime.cleanup["scene_context_closed"]
    if query == 35:
        assert result["executed"] == 8 and len(approved) == 2 and not approved[1]["receipt"]["repaired"], result
        assert category == "diagnostic_complete"
    else:
        assert result["executed"] == 4 and len(approved) == 1 and result["failure"]["code"] == "position_limit", result
        assert category == "controlled_candidate_stop" and runtime.candidate_stop["no_physics_advance"], result
        assert backend.tick == sent[1]["stamp"]["tick"]
    summary = dict(query=query, old_approved=old_count, projected_executed=4,
                   projection_m=corrections[0]["left_task_translation_m"],
                   projection_attempts=len(corrections[0]["attempts"]),
                   second_request_tick=sent[1]["stamp"]["tick"], first_request_tick=sent[0]["stamp"]["tick"],
                   total_physics_steps=backend.tick - sent[0]["stamp"]["tick"], queries=len(sent),
                   fresh_state_changed=True, discarded_rows=46,
                   second_response_approved=len(approved) == 2, total_executed=result["executed"],
                   classification=category, failure=result["failure"], cleanup=runtime.cleanup,
                   candidate_stop=runtime.candidate_stop)
    print("CPU_ONLINE", query, category, result["executed"], flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "assets", "job", "snapshots", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    data = load(args.snapshots)
    source = args.job / "trials" / EPISODE / "episode"
    for name in ("events.jsonl", "result.json", "action-00035.npy", "action-00036.npy"):
        assert digest(source / name) == data["source_sha256"][f"trials/{EPISODE}/episode/{name}"]
    config = load(args.release / "configs" / f"{EPISODE}.json")
    assert config["task_tcp_projection"] == METHOD
    validate_config(config)
    row, = [row for row in load(args.release / "unified-plan.json")["trials"] if row["episode_id"] == EPISODE]
    args.output.mkdir()
    summaries = []
    for query in (35, 36):
        reference = data["requests"][str(query)]
        chunk = np.load(source / f"action-{query:05d}.npy", allow_pickle=False)
        summaries.append(replay(query, args, config, row, data["initial"], reference, chunk))
        save(args.output / "summary.json", dict(
            scope="actual CPU PhysX joints, fingers, block and target; fixture RGB; loopback WebSocket policy substitute",
            frozen_model_calls=0, gpu_jobs_submitted=0,
            snapshot_evidence_sha256=sha256(args.snapshots.read_bytes()).hexdigest(), reports=summaries,
        ))


if __name__ == "__main__":
    main()
