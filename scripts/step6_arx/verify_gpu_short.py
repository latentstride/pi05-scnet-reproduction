"""Check the real RGB, MP4 and first-model-query evidence for one short trial."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .checkpoint import read_json
from .protocol import SHAPE, actions_from, decode, validate_metadata, validate_request


def verify(trial):
    from imageio_ffmpeg import read_frames

    receipt = read_json(trial / "receipt.json")
    campaign = read_json(trial.parent / "campaign.json")
    if (campaign.get("acceptance_only") is not True or receipt.get("acceptance_only") is not True
            or receipt["source"] != "step5_model" or receipt["queries"] != 1
            or receipt["rgb_status"] != "OBSERVED" or receipt["cleanup_errors"]
            or receipt["outcome"] not in {"ACTION_LIMIT", "TASK_SUCCESS", "SAFETY_STOP"}
            or receipt["policy_rows_attempted"] > campaign["plan"]["execute_cap"]
            or receipt["video"] != "head.mp4"
            or receipt["variant"] == "lora" and receipt["pair_match"] is not True):
        raise ValueError("incomplete real short-window receipt")
    if receipt["outcome"] == "SAFETY_STOP" and (
            receipt["failure"]["phase"] != "safe_execution"
            or receipt["failure"]["diagnostic"].get("geometry") == "actual"
            or receipt["failure"]["status"] == "backend_error"):
        raise ValueError("unsafe measured execution or invalid stop phase")

    events = {}
    for line in (trial / "trajectory.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["kind"] in ("policy_handshake", "policy_request", "policy_response"):
            if event["kind"] in events:
                raise ValueError("extra WebSocket exchange")
            events[event["kind"]] = event
    messages = {}
    for kind, event in events.items():
        ref = event["payload"]
        data = (trial / ref["file"]).read_bytes()
        if sha256(data).hexdigest() != ref["sha256"] or len(data) != ref["nbytes"]:
            raise ValueError(f"wire capture differs from {kind} receipt")
        messages[kind] = decode(data)
    variant = receipt["variant"]
    validate_metadata(messages["policy_handshake"], variant, campaign["plan"], fixture=False)
    if messages["policy_handshake"] != receipt["checkpoint"]:
        raise ValueError("model identity differs from the wire handshake")
    request = messages["policy_request"]
    validate_request(request, 0)
    actions = actions_from(messages["policy_response"])
    request_event, response_event = events["policy_request"], events["policy_response"]
    if (request["query_index"] != 0 or request_event["stamp"]["tick"] != 0
            or request_event["stamp"] != response_event["stamp"] or request_event["rgb_source"] != "OBSERVED"):
        raise ValueError("first response is not bound to the rendered reset frame")
    cameras = {}
    for name, img in request["images"].items():
        if not np.any(img != img[0, 0]):
            raise ValueError(f"uniform image: {name}")
        cameras[name] = dict(shape=list(img.shape), sha256=sha256(img.tobytes()).hexdigest())
    if len({cam["sha256"] for cam in cameras.values()}) != 3:
        raise ValueError("different cameras returned the same image")

    frames = read_frames(str(trial / receipt["video"]), pix_fmt="rgb24")
    try:
        meta = next(frames)
        if tuple(meta["size"]) != (SHAPE[1], SHAPE[0]):
            raise ValueError("video dimensions differ from camera")
        frame_bytes = int(np.prod(SHAPE))
        count = sum(len(frame) == frame_bytes for frame in frames)
        if count < 2:
            raise ValueError("video needs at least two decodable frames")
    finally:
        frames.close()

    return dict(status="PASS", variant=variant, cameras=cameras, decoded_video_frames=count,
                action_shape=list(actions.shape), physics_ticks=receipt["physics_ticks"],
                executed_actions=receipt["policy_actions"], execution_outcome=receipt["outcome"],
                initial_pair_match=receipt["pair_match"], checkpoint=receipt["checkpoint"],
                first_request_response_s=(response_event["unix_ns"] - request_event["unix_ns"]) / 1e9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.trial), indent=2))


if __name__ == "__main__":
    main()
