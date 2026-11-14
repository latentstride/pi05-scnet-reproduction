import hashlib, json, sys, collections
from pathlib import Path
import numpy as np
from analyze_g2_arx_result import analyze, jsonl, collision_pairs

root = Path(sys.argv[1])
out = Path(sys.argv[2])
ep = "unified-seed1-arx-full-w4"
labels = dict(G1="127938162", G2="127931687", G3="127931701", G4="127938303")
sacct = {
    r.split("|")[0]: r.split("|")[1:]
    for r in (root / "slurm.psv").read_text().splitlines()
}
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = {}
for label, j in labels.items():
    p = root / j
    tf = p / "trials" / ep
    folder = tf / "episode"
    a = analyze(p, ep)
    activity = a["activity"]
    qs = activity["queries"]
    result = read(tf / "receipt.json")
    failure = result["result"]["failure"]
    diag = failure["diagnostic"]
    attempts = diag.get("attempts", [])
    phases = [
        r for _, r in jsonl(p / "model/phases.jsonl") if r and r["event"] == "end"
    ]
    auth = read(p / "authorization-receipt.json")
    requests = []
    positions = []
    tool_distance = []
    finger_positions = []
    measured_steps = 0
    for _, e in jsonl(folder / "events.jsonl"):
        if not e:
            continue
        if e.get("kind") == "policy_request":
            requests.append(e)
        if e.get("physics_step_count"):
            measured_steps += e["physics_step_count"]
            pose = (e.get("contact_state") or {}).get("block_pose_task")
            if pose is not None:
                xyz = np.asarray(pose)[:3, 3].tolist()
                positions.append((e["tick"], xyz))
                tools = np.asarray(e["actual_tools_task"])[:, :3, 3]
                tool_distance.append(np.linalg.norm(tools - xyz, axis=1))
                finger_positions.append(e["actual_fingers"])
    xyz = np.asarray([x[1] for x in positions])
    prefix = []
    for q in qs:
        f = folder / f'action-{q["query"]:05d}.npy'
        if not f.is_file():
            prefix.append(None)
        else:
            x = np.load(f, allow_pickle=False)
            assert x.shape == (50, 14)
            prefix.append(
                dict(
                    array_hash=hashlib.sha256(x.tobytes()).hexdigest(),
                    closed_by_w4=[bool((x[:4, i] <= 0.5).any()) for i in (6, 13)],
                    closed_any=[bool((x[:, i] <= 0.5).any()) for i in (6, 13)],
                )
            )
    assert (
        a["counters"]["result_queries"] == len(qs)
        and measured_steps == activity["physical_steps"]
    )
    rows[label] = dict(
        job_id=j,
        slurm=dict(
            zip(
                (
                    "state",
                    "exit",
                    "start_beijing",
                    "end_beijing",
                    "elapsed",
                    "node",
                    "alloc_tres",
                ),
                sacct[j],
            )
        ),
        authorization={
            k: auth.get(k)
            for k in (
                "archive_sha256",
                "configuration_sha256",
                "commit",
                "yuan_per_gpu_hour",
                "max_cost_yuan",
            )
        },
        rng_reset=read(tf / "rng-reset.json"),
        startup=a["startup"],
        model_phase_seconds={
            r["stage"]: r["elapsed_s"] for r in phases if r["stage"] != "inference"
        },
        candidate_stop=result.get("candidate_stop"),
        formal_score_eligible=result["result"].get("formal_score_eligible"),
        measured_min_tcp_to_block_center_m=(
            np.min(tool_distance, axis=0).tolist() if tool_distance else None
        ),
        measured_min_finger_joints_m=(
            np.min(finger_positions, axis=0).tolist() if finger_positions else None
        ),
        counters=a["counters"],
        termination=a["termination"],
        event_counts=activity["counts"],
        physical_steps=measured_steps,
        contact_phase_ticks=activity["contact_phase_ticks"],
        finger_contact_ticks=activity["finger_contact_ticks"],
        holder_ticks=activity["holder_ticks"],
        stages=activity["stages"],
        last_measured=activity["last_measured"],
        block_task_m=(
            dict(
                first=positions[0],
                last=positions[-1],
                min=xyz.min(axis=0).tolist(),
                max=xyz.max(axis=0).tolist(),
                samples_with_pose=len(positions),
            )
            if positions
            else None
        ),
        projection_plans=activity["projection_plans"],
        projection_fresh_requests=[q for q in qs if q.get("projection_reobservation")],
        terminal_candidates=[
            dict(
                status=x.get("status"),
                detail=x.get("detail"),
                accepted=x.get("accepted"),
                translation=x.get("left_task_translation_m"),
                link_pairs=collision_pairs(x),
            )
            for x in attempts
        ],
        collision_audits=a["collision_audits"],
        terminal_rejections=activity["rejection_events"][-6:],
        first_request_stamp=requests[0]["stamp"],
        first_request_state=requests[0].get("state"),
        gripper_prediction=dict(
            blocks_with_le_0_5_any=[
                sum(x["closed_any"][i] for x in prefix if x) for i in (0, 1)
            ],
            blocks_with_le_0_5_in_w4=[
                sum(x["closed_by_w4"][i] for x in prefix if x) for i in (0, 1)
            ],
            saved_blocks=sum(x is not None for x in prefix),
        ),
        action_hashes=[x["array_hash"] if x else None for x in prefix],
        evidence_sha256={
            str(f.relative_to(p)): sha(f) for f in sorted(p.rglob("*")) if f.is_file()
        },
    )
a, b = rows["G1"], rows["G4"]
pair = dict(
    initial_readback_byte_equal=a["evidence_sha256"][
        f"trials/{ep}/episode/initial-readback.json"
    ]
    == b["evidence_sha256"][f"trials/{ep}/episode/initial-readback.json"],
    first_request_state_equal=a["first_request_state"] == b["first_request_state"],
    first_different_action_chunk=next(
        (
            i
            for i, (x, y) in enumerate(zip(a["action_hashes"], b["action_hashes"]), 1)
            if x != y
        ),
        None,
    ),
)
out.write_text(
    json.dumps(
        dict(trials=rows, matched_seed_pair=pair),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    + "\n"
)
