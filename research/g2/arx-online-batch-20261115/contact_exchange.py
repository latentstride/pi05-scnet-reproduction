"""Exercise the production direct controller and wire contract on a CPU PhysX snapshot."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Thread
from time import monotonic
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

from g2_adapter import deadline
from g2_arx_direct import DirectAdapter, DirectController
from g2_arx_endpoint_cpu import build_scene, save
from g2_arx_transfer_probe_cpu import restart_backend
from pace_runtime import TASK, open_policy, validate_request, validate_response


def run(args):
    raw = args.input.read_bytes()
    spec = json.loads(raw)
    config = spec["config"]
    args.output.mkdir()
    scene, old, snapshot, drive = build_scene(spec, args.audit, args.assets, args.output)
    backend, _, _ = restart_backend(scene, old, snapshot, drive, config, spec["initial"], args.assets)
    actors = {actor.name: actor for actor in scene.get_all_actors()}
    backend.env.table, backend.env.target_box = actors["table"], actors["target_box"]

    def fixture_rgb():
        return {"observation": {
            camera: {"rgb": np.full((8, 8, 3), (backend.tick + offset) % 256, np.uint8)}
            for offset, camera in enumerate(("head_camera", "left_camera", "right_camera"))
        }}

    backend.env.get_obs = fixture_rgb
    backend.env.check_success = lambda: False
    backend.video = SimpleNamespace(frame=lambda _: None)
    adapter = DirectAdapter(
        old.aloha, old.arx, old.source_frames, np.eye(4), old.seed, old.source_speed,
        target_collision=old.target_collision, motion=old.motion.motion,
        grip_rule=old.grip_rule, grip_max_clip=old.grip_max_clip,
    )
    controller = DirectController(adapter, backend)
    started = monotonic()
    check = deadline(180)

    def remaining():
        check()
        return 180 - (monotonic() - started)

    observations, sent, planned, boundaries = [], [], [], []
    requests = []
    archived_action = np.asarray(spec["chunk"], dtype=np.float64)[args.first_row - 1]
    # The remaining 49 rows are sentinels. A second plan must originate from the second reply.
    chunk = np.full((50, 14), 8.0)
    chunk[0] = archived_action

    def handler(socket):
        socket.send(msgpack_numpy.packb(dict(backend="checkpoint", task=TASK, protocol=1)))
        for index in range(2):
            request = msgpack_numpy.unpackb(socket.recv(timeout=30))
            validate_request(request)
            sent.append(request)
            reply = chunk if index == 0 else np.repeat(request["state"][None, :], 50, axis=0)
            response = dict(actions=reply)
            validate_response(response)
            socket.send(msgpack_numpy.packb(response))

    def log(event):
        kind = event.get("kind")
        if kind == "policy_request":
            snapshot, request = event["snapshot"], event["request"]
            requests.append(request)
            observations.append(dict(stamp=asdict(snapshot.stamp),
                state=request["state"], prompt=request["prompt"],
                actual=[asdict(arm) for arm in snapshot.arms], objects_task=snapshot.objects_task,
                image_sha256={name: sha256(rgb.tobytes()).hexdigest() for name, rgb in request["images"].items()},
                image_shapes={name: list(rgb.shape) for name, rgb in request["images"].items()}))
        elif kind == "direct_prefix_complete":
            boundaries.append(event)
        elif event.get("physics_step_count"):
            physics.write(json.dumps(event, default=lambda data: data.tolist(), allow_nan=False) + "\n")

    policy_row = adapter.policy_row

    def record_plan(action, snapshot, remaining_s):
        planned.append(dict(tick=snapshot.stamp.tick, action=np.asarray(action).tolist()))
        return policy_row(action, snapshot, remaining_s)

    adapter.policy_row = record_plan
    start = backend.tick
    with (args.output / "physics.jsonl").open("x") as physics:
        with serve(handler, "127.0.0.1", 0, compression=None) as server:
            runner = Thread(target=server.serve_forever, daemon=True)
            runner.start()
            try:
                port = server.socket.getsockname()[1]
                with open_policy(f"ws://127.0.0.1:{port}", "checkpoint", 10) as policy:
                    controller.cycle(policy, config["prompt"], 50, remaining, log)
                    contact = backend.collision.tracker.snapshot()
                    assert boundaries[0]["reason"] == "measured_first_contact", boundaries
                    assert boundaries[0]["executed_actions"] == 1 and boundaries[0]["discarded_rows"] == 49
                    assert any(contact["contact_fingers"].values()), contact
                    controller.cycle(policy, config["prompt"], 1, remaining, log)
            finally:
                server.shutdown()
                runner.join(timeout=10)
                before_hold = backend.tick
                backend.hold()

    assert len(sent) == len(observations) == len(boundaries) == 2
    assert controller.policy_actions == 2
    assert backend.tick == before_hold > observations[1]["stamp"]["tick"] > start
    assert observations[1]["stamp"] == boundaries[0]["stamp"]
    assert observations[0]["image_sha256"] != observations[1]["image_sha256"]
    assert not np.array_equal(sent[0]["state"], sent[1]["state"])
    for received, request in zip(sent, requests):
        assert received["prompt"] == request["prompt"]
        np.testing.assert_array_equal(received["state"], request["state"])
        for name in received["images"]:
            np.testing.assert_array_equal(received["images"][name], request["images"][name])
    resumed = [row for row in planned if row["tick"] >= observations[1]["stamp"]["tick"]]
    assert resumed[0]["tick"] == observations[1]["stamp"]["tick"]
    for row in resumed:
        np.testing.assert_array_equal(row["action"], sent[1]["state"])
    save(args.output / "receipt.json", dict(
        status="PASS", input_sha256=sha256(raw).hexdigest(), archive_policy_row=args.first_row,
        frozen_policy_inference=False, rgb="distinct per-camera CPU RGB fixtures",
        second_response="measured source-coordinate hold for interface validation",
        start_tick=start, stop_tick=backend.tick, observations=observations,
        boundaries=boundaries, contact_at_requery=contact, plans=planned,
        physically_executed_policy_rows=controller.policy_actions, hold_same_tick=True,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("input", "audit", "assets", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--first-row", type=int, required=True)
    run(parser.parse_args())
