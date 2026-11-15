"""Audit the unmodified first policy chunk and its physical readback. No inference."""
import argparse
from collections import defaultdict
from csv import DictReader
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

from g2_arx_grasp_adaptation import load_jaws  # Mesh reader only, no grasp adaptation.
from g2_assets import load_arms
from g2_initial import matrix7
from g2_kinematics import pose_error
from robotwin_control_geometry import inverse_pose
from audit import configs

from geometry import closure_rows, finger_hits, fk, relative
from input_semantics import CAMERAS, ROOT, decode, image_semantics, policy_contract, read, rebuild_states

STUDY = ROOT / "research/g2/arx-online-batch-20261115"
ALOHA_STUDY = ROOT / "research/g2/aloha-six-20261111"
ALOHA = ("127589584", "127589587", "127589590")
HALF = np.array([.03, .03, .10])


def verified(path, root, hashes):
    with path.open("rb") as stream:
        actual = file_digest(stream, "sha256").hexdigest()
    if hashes[str(path.relative_to(root))] != actual:
        raise ValueError(f"archive hash differs: {path.name}")
    return actual


def load_array(path):
    return np.load(path, allow_pickle=False)


def save(path, data):
    path.write_text(json.dumps(data, allow_nan=False, indent=2, default=lambda x: x.tolist()) + "\n")


def poses(batch, frames, arms, jaws, block, actual):
    rows = []
    for number, flat in enumerate(batch, 1):
        arr = flat.reshape(2, 7)
        pose = fk(arms, frames, arr[:, :6])
        grip = arr[:, 6].clip(0, 1)
        closed = np.array([jaw.center([g, g]) for jaw, g in zip(jaws, grip)])
        left = dict(initial_fingers=finger_hits(jaws[0], pose[0], block, actual[0], HALF),
                    requested_fingers=finger_hits(jaws[0], pose[0], block, [grip[0]]*2, HALF))
        rows.append(dict(row=number, raw_action=flat, tcp_task=pose, tool_geometry=relative(pose, block, closed),
                         grasp_center_tcp_m=closed, left_mesh_intersection=left, gripper_request=arr[:, 6]))
    return rows


def geometry_at(tools, block, jaws, openness):
    centers = [jaw.center(amount) for jaw, amount in zip(jaws, openness)]
    return relative(tools, block, np.asarray(centers))


def selected(rows, chunk):
    limits = closure_rows(chunk)
    hits = [r for r in rows if any(r["left_mesh_intersection"]["initial_fingers"])]
    one = [r["row"] for r in hits if sum(r["left_mesh_intersection"]["initial_fingers"]) == 1]
    return dict(first_requests_le=limits, min_left_request=float(chunk[:, 6].min()),
                first_single_finger_endpoint=one[0] if one else None, single_finger_endpoints=one,
                first_mesh_endpoint=hits[0]["row"] if hits else None)


def xyz(pose):
    return 1000*np.asarray(pose)[:3, 3]


def initial_geometry(block, target, frames, tools, jaws, fingers, source_frames):
    return dict(block_task=block, placement_task=target,
                block_pose_in_physical_bases=[inverse_pose(x) @ block for x in frames],
                block_pose_in_virtual_bases=[inverse_pose(x) @ block for x in source_frames],
                block_to_placement_mm=xyz(inverse_pose(block) @ target),
                block_to_placement_xy_mm=float(np.linalg.norm((xyz(target) - xyz(block))[:2])),
                tcp_task=tools, **geometry_at(tools, block, jaws, fingers))


def event_stream(path):
    with path.open() as stream:
        for line, payload in enumerate(stream, 1):
            yield line, json.loads(payload)


def arx_trace(path, arms, jaws):
    requests, events, plans = [], {}, defaultdict(list)
    starts, movement = None, {}
    row_max = {}
    row = 0
    for line, event in event_stream(path):
        kind = event["kind"]
        q = len(requests)
        if kind == "policy_request":
            if len(requests) == 2:
                break
            requests.append(dict(event, event_line=line))
            if starts is None:
                starts = np.asarray(event["objects_task"]["block"])
            continue
        if kind == "direct_row_planning":
            row = event["row_in_prefix"]
        if q == 1 and kind == "approved_plan":
            plans[row].append((line, event["receipt"]))
        if event.get("physics_step_count") and event.get("contact_state"):
            state = event["contact_state"]
            block = np.asarray(state["block_pose_task"])
            shift = xyz(block) - xyz(starts)
            fingers = np.asarray(event["actual_fingers"])
            openness = [np.clip((values - arm.grip.closed) / jaw.travel_m, 0, 1)
                        for arm, jaw, values in zip(arms, jaws, fingers)]
            actual_g = [arm.grip.read(values) for arm, values in zip(arms, fingers)]
            planar = float(np.linalg.norm(shift[:2]))
            summary = dict(line=line, query=q, row=row, tick=event["tick"], sim_s=event["sim_s"],
                           wall_elapsed_s=event["elapsed_s"], block_xyz_task_mm=xyz(block),
                           block_xy_shift_mm=planar, left_aperture=actual_g[0],
                           contacts=state["contact_fingers"])
            key = (q, row)
            if key not in row_max or planar > row_max[key]["block_xy_shift_mm"]:
                row_max[key] = summary
            if "first_single" not in events and not state["contact_fingers"]["left"]:
                movement["max_xy_strictly_before_first_contact_mm"] = max(
                    planar, movement.get("max_xy_strictly_before_first_contact_mm", 0.))
            if "first_actual_aperture_le_095" not in events:
                if planar > movement.get("max_xy_before_actual_closure_mm", -1):
                    movement.update(max_xy_before_actual_closure_mm=planar, max_shift_sample=summary)
                if actual_g[0] <= .95:
                    events["first_actual_aperture_le_095"] = summary
            for threshold in (1, 5):
                if planar > threshold:
                    events.setdefault(f"first_shift_gt_{threshold}mm", summary)
            fingers_left = len(state["contact_fingers"]["left"])
            for count, name in ((1, "first_single"), (2, "first_bilateral")):
                if fingers_left == count and name not in events:
                    actual = np.asarray(event["actual_tools_task"])
                    command = np.asarray(event["target_tools_task"])
                    events[name] = dict(summary, physical_contacts=event.get("physical_contacts"),
                                        instantaneous_command_geometry=geometry_at(command, block, jaws, openness),
                                        position_error_mm=1000*np.asarray(event["position_error_m"]),
                                        actual_geometry=geometry_at(actual, block, jaws, openness))
            if q == 1:
                movement["q1_last_step"] = summary
    return requests, plans, events, dict(movement, per_row_max_xy=list(row_max.values()))


def contact_comparison(rows, following, trace, block_q1, block_q2):
    first = trace.get("first_single")
    if first is None:
        return None
    assert first["query"] == 1
    at_contact = rows[first["row"]-1]
    first_y = at_contact["tool_geometry"]["block_xyz_jaw_mm"][0, 1]
    q2 = {}
    for name, item in following.items():
        if item is None:
            continue
        same_pose = relative(item["tcp_task"], block_q1, item["grasp_center_tcp_m"])["block_xyz_jaw_mm"][0, 1]
        actual = item["tool_geometry"]["block_xyz_jaw_mm"][0, 1]
        q2[name] = dict(row=item["row"], signed_y_mm=actual, at_q1_block_pose_y_mm=same_pose,
                        new_target_yaw_position_contribution_mm=same_pose - first_y,
                        block_translation_contribution_mm=actual-same_pose, total_difference_mm=actual-first_y)
    return dict(q1_contact_row=first["row"], q1_policy_left_grip=at_contact["gripper_request"][0],
                q1_signed_goal_y_mm=first_y,
                q1_actual_signed_y_mm=first["actual_geometry"]["block_xyz_jaw_mm"][0, 1],
                q2_block_shift_task_xyz_mm=xyz(block_q2)-xyz(block_q1), q2=q2)


def candidate_fk(plans, rows, arms, frames, jaws, block, failure):
    counts = 0
    errors = []
    def check(goal, q, reason):
        mapped = fk(arms, frames, q)
        e = 1000 * np.linalg.norm(goal[:, :3, 3]-mapped[:, :3, 3], axis=1)
        return dict(position_residual_mm=e, tcp_task=mapped, kind=reason,
                    lateral_change_mm=(relative(mapped, block, np.asarray([j.center() for j in jaws]))[
                        "block_xyz_jaw_mm"] - relative(goal, block, np.asarray([j.center() for j in jaws]))[
                        "block_xyz_jaw_mm"])[:, 1])

    for row, items in plans.items():
        goal = rows[row-1]["tcp_task"]
        comparisons = []
        for line, detail in items:
            counts += 1
            req = (detail.get("requests") or [detail["original_request"]])[-1]
            np.testing.assert_allclose(goal, req["task_tools"], rtol=0, atol=1e-9)
            q = req.get("ik_target", detail.get("original_ik_target"))
            if q is not None:
                record = check(goal, q, "archived_ik_endpoint")
                errors.append(record["position_residual_mm"])
                comparisons.append(dict(record, plan_event_line=line))
        if comparisons:
            rows[row-1]["arx_ik"] = dict(first=comparisons[0], last=comparisons[-1], segments=len(comparisons))
    candidates = []
    if not plans:
        for item in failure.get("diagnostic", {}).get("candidate_verdicts", ()):
            candidates.append(check(rows[0]["tcp_task"], item["candidate_q"], "rejected_ik_candidate"))
    return dict(approved_goal_comparisons=counts,
                approved_ik_max_mm=np.array(errors).max(axis=0) if errors else None,
                unexecuted_rejected_first_row=candidates)


def arx_trial(label, entry, args, members, policy, fovy):
    root = args.arx
    folder = root / entry["result_directory"]
    trial = folder / "trials" / entry["episode"]
    data = trial / "episode"
    hashes = read(STUDY / "results" / label / "members.json")
    files = [trial / "receipt.json", trial / "rng-reset.json", folder / "model/startup.json", data / "events.jsonl",
             data / "initial-rgb.npz"]
    checked = {path.name: verified(path, root, hashes) for path in files}
    result = read(trial / "receipt.json")["result"]
    cfg = configs(label, entry["episode"])
    source, target = (load_arms(args.assets, args.asset_audit, robot, cfg["calibration"][robot])
                      for robot in ("aloha-agilex", "ARX-X5"))
    source_frames = [inverse_pose(matrix7(cfg["source_world_task"])) @ matrix7(p) for p in cfg["source_base_world"]]
    frames = [inverse_pose(matrix7(cfg["world_task"])) @ matrix7(result["initial"]["robot_base_world"][side])
              for side in ("left", "right")]
    jaws = [load_jaws(args.assets / members["ARX-X5"]["urdf_member"], arm) for arm in target]
    source_jaws = [load_jaws(args.assets / members["aloha-agilex"]["urdf_member"], arm) for arm in source]
    requests, plans, trace, movement = arx_trace(data / "events.jsonl", target, jaws)
    inputs, blocks, encoded = [], [], []
    arches = []
    for i, req in enumerate(requests, 1):
        assert req["prompt"] == cfg["prompt"]
        obs, act = data / f"observation-{i:05}.npz", data / f"action-{i:05}.npy"
        for p in (obs, act):
            checked[p.name] = verified(p, root, hashes)
        with load_array(obs) as archive:
            inputs.append(dict(archive))
        block = load_array(act)
        assert block.shape == (50, 14) and np.isfinite(block).all()
        blocks.append(block)
        req["state"] = np.asarray(req["state"], dtype=np.float32)
        encoded.append(decode(policy, req["state"], block))
        world_task = np.asarray(req["world_task"])
        arches.append(image_semantics(policy, req, inputs[-1], req["actual"], args.assets, members["ARX-X5"],
                                     True, world_task @ np.asarray(req["objects_task"]["block"]), fovy))
    assert checked["initial-rgb.npz"] == checked["observation-00001.npz"]
    state = rebuild_states(source, target, cfg, requests, inputs)
    rng = read(trial / "rng-reset.json")
    startup = read(folder / "model/startup.json")
    assert rng["seed"] == cfg["policy_rng_seed"]
    assert startup["checkpoint_revision"] == policy.checkpoint["revision"]
    expected_files = {entry["path"]: (entry["bytes"], entry["sha256"]) for entry in policy.checkpoint["files"]}
    assert {name: (value[0], value[2]) for name,value in startup["checkpoint_files"].items()} == expected_files
    norm = policy.checkpoint["norm_stats"]
    assert startup["checkpoint_files"][norm][2] == policy.checkpoint["norm_stats_sha256"]
    q1 = requests[0]
    block = np.asarray(q1["objects_task"]["block"])
    q1_tools = fk(target, frames, state[0]["actual_arms_rad"])
    fractions = [(actual-arm.grip.closed)/jaw.travel_m for actual, arm, jaw in
                 zip(state[0]["actual_fingers_m"], target, jaws)]
    initial = initial_geometry(block, np.asarray(q1["objects_task"]["target"]), frames,
                               q1_tools, jaws, fractions, source_frames)
    rows = poses(blocks[0], source_frames, source, jaws, block, fractions)
    source_fractions = np.repeat(q1["state"].reshape(2, 7)[:, 6, None], 2, axis=1)
    for row in rows:
        g = np.clip(row["gripper_request"], 0, 1)
        row["native_arx_finger_targets_mm"] = 1000*np.array([arm.grip.target(x) for arm,x in zip(target,g)])
        row["virtual_aloha_finger_targets_mm"] = 1000*np.array([arm.grip.target(x) for arm,x in zip(source,g)])
        row["same_raw_tcp_aloha_jaw_y_mm"] = float(row["tool_geometry"]["block_xyz_tcp_mm"][0, 1]
                                                 - 1000*source_jaws[0].center([g[0]]*2)[1])
        row["same_raw_tcp_aloha_left_mesh"] = finger_hits(source_jaws[0], row["tcp_task"][0],
                                                          block, source_fractions[0], HALF)
    info = selected(rows, blocks[0])
    mapping = candidate_fk(plans, rows, target, frames, jaws, block, result.get("failure", {}))
    following = None
    if len(requests) > 1:
        next_block = np.asarray(requests[1]["objects_task"]["block"])
        q2_rows = poses(blocks[1], source_frames, source, jaws, next_block, fractions)
        close = closure_rows(blocks[1])["0.5"]
        following = contact_comparison(rows, {"first_row": q2_rows[0], "effective_close":
                    q2_rows[close-1] if close else None}, trace, block, next_block)
    initial_proxy = fk(source, source_frames, q1["state"].reshape(2, 7)[:, :6])
    # Check only what these baselines can establish at this pose, not future path reachability.
    identity = dict(job=entry["job_id"], rng=rng, checkpoint=startup["checkpoint_revision"],
                    norm_stats_sha256=startup["checkpoint_files"][norm][2], checkpoint_file_matches=len(expected_files),
                    scene_seed=cfg["scene_seed"])
    first = dict(identity=identity, request=q1, rgb=arches[0], verified_files_sha256=checked,
                 initial=initial, virtual_initial_tcp_task=initial_proxy, rows=rows,
                 summary=info, q2_bias_comparison=following, physical=dict(events=trace, **movement),
                 execution=mapping, terminal_failure=result["failure"])
    semantics = dict(identity=identity, state=state, decode=encoded, cameras=arches, prompt=q1["prompt"])
    return first, semantics, inputs[0]


def aloha_trial(job, args, members, source_config, policy, fovy):
    episode, = args.aloha.glob(f"results/pace-g2-parallel/*/{job}/trials/*")
    hashes = read(ALOHA_STUDY / "raw-manifest.json")
    required = ("events.jsonl", "physics.jsonl", "request-00.npz", "request-01.npz",
                "chunk-00.npy", "chunk-01.npy", "receipt.json")
    checked = {name: verified(episode / name, args.aloha, hashes) for name in required}
    receipt = read(episode / "receipt.json")
    assert receipt["success"] and receipt["selector"]["horizon"] == 50
    arms = load_arms(args.assets, args.asset_audit, "aloha-agilex", source_config["calibration"]["aloha-agilex"])
    jaws = [load_jaws(args.assets / members["aloha-agilex"]["urdf_member"], a) for a in arms]
    q2_tick = None
    rng = {}
    for line, event in event_stream(episode / "events.jsonl"):
        if event["kind"] == "rng_reset":
            rng = event["receipt"]
        if event["kind"] == "response" and event["query"] == 0:
            rng = dict(rng, sample=event["rng"]["sample"], first_response_line=line)
        if event["kind"] == "action_complete" and event["action"] == 50:
            q2_tick = event["tick"]
            break
    assert q2_tick is not None
    history, contacts = {}, {}
    max_before_close = 0.
    for line, e in event_stream(episode / "physics.jsonl"):
        if e["tick"] in (0, q2_tick):
            history[e["tick"]] = e
        first = history[0]
        names = first["joint_names"]
        block = np.asarray(e["block_pose"])
        native = [np.asarray(e["qpos"])[[names.index(n) for n in arm.grip.names]] for arm in arms]
        actual = [arm.grip.read(g) for arm, g in zip(arms, native)]
        planar = np.linalg.norm((xyz(block) - xyz(first["block_pose"]))[:2])
        if "first_actual_aperture_le_095" not in contacts:
            max_before_close = max(max_before_close, planar)
            if actual[0] <= .95:
                contacts["first_actual_aperture_le_095"] = dict(tick=e["tick"], action=e["action"], shift_mm=planar)
        seen = set()
        for hit in e["contacts"]:
            if "box" in hit["pair"] and any(np.linalg.norm(p["impulse"])>=1.e-8 for p in hit["points"]):
                seen.update(set(hit["pair"]) & {"fl_link7", "fl_link8"})
        name = {1:"first_single", 2:"first_bilateral"}.get(len(seen))
        if name is not None and name not in contacts:
            joint = np.asarray(e["qpos"])
            flange_q = [joint[[names.index(n) for n in arm.chain.names]] for arm in arms]
            tools = fk(arms, [np.asarray(e["robot_root"])]*2, flange_q)
            fk_errors = [np.linalg.norm(tool[:3,3] - matrix7(tcp)[:3,3])
                         for tool,tcp in zip(tools,e["tcp"])]
            np.testing.assert_allclose(fk_errors, 0., atol=2e-6)
            openness = [(value-arm.grip.closed)/jaw.travel_m for value,arm,jaw in zip(native,arms,jaws)]
            contacts[name] = dict(line=line, tick=e["tick"], sim_s=e["tick"]*first["step_period_s"],
                                  action=e["action"], signed_y_mm=geometry_at(tools, block, jaws, openness)
                                  ["block_xyz_jaw_mm"][0, 1], aperture_actual=actual[0],
                                  aperture_command=e["normalized_gripper_command"][0],
                                  block_xy_shift_mm=planar, fingers=sorted(seen))
        if e["tick"] >= q2_tick and "first_bilateral" in contacts:
            break
    observations, inputs, rows_by_query, closure = [], [], [], []
    for q, tick in enumerate((0, q2_tick)):
        e = history[tick]
        names = history[0]["joint_names"]
        joints = np.asarray(e["qpos"])
        physical = [dict(world_base=e["robot_root"], names=names, qpos=joints)] * 2
        arm_q = np.array([joints[[names.index(n) for n in arm.chain.names]] for arm in arms])
        fingers = [joints[[names.index(n) for n in arm.grip.names]] for arm in arms]
        openness = [(value-arm.grip.closed)/jaw.travel_m for value,arm,jaw in zip(fingers,arms,jaws)]
        frames = [np.asarray(e["robot_root"])]*2
        tools = fk(arms, frames, arm_q)
        with load_array(episode / f"request-{q:02d}.npz") as raw:
            images = {key: raw[key] for key in CAMERAS}
            state = raw["state"]
            prompt = str(raw["prompt"])
        assert prompt == receipt["prompt"]
        inputs.append(images)
        req = dict(state=state, prompt=prompt)
        block = np.asarray(e["block_pose"])
        initial = initial_geometry(block, np.asarray(history[0]["target_pose"]), frames, tools, jaws, openness, frames)
        chunk = load_array(episode / f"chunk-{q:02d}.npy")
        assert chunk.shape == (50, 14) and np.isfinite(chunk).all()
        rows = poses(chunk, frames, arms, jaws, block, openness)
        rows_by_query.append(rows)
        limits = selected(rows, chunk)
        close = limits["first_requests_le"]["0.5"]
        closure.append(dict(limits, effective_close=rows[close-1] if close else None))
        observations.append(dict(query=q+1, tick=tick, state=state, state_dtype=str(state.dtype), prompt=prompt,
                                 task_geometry=initial, measured_joint_angles_rad=arm_q,
                                 drive_target_minus_measured_rad=state.reshape(2,7)[:, :6]-arm_q,
                                 decode=decode(policy, state, chunk),
                                 camera=image_semantics(policy, req, images, physical, args.assets,
                                     members["aloha-agilex"], False, block, fovy)))
    first = contacts.get("first_single")
    def goal_at(event):
        one_based = event["action"]
        index, row = divmod(one_based-1, 50)
        return dict(query=index+1, **rows_by_query[index][row]) if index < len(rows_by_query) else None
    native_contact_goal = goal_at(first) if first else None
    summary = read(ALOHA_STUDY / "summary.json")
    locked = next(r for r in summary["rows"] if r["job_id"] == job)
    return (dict(job=job, scene=receipt["scene_seed"], rng=rng, observations=observations, closure=closure,
                q1_rows=rows_by_query[0], first_contact_target=native_contact_goal, contacts=contacts,
                max_xy_before_actual_closure_mm=max_before_close, original_sha256=checked,
                provenance=dict(config_sha256=locked["config_sha256"], archive_sha256=locked["archive_sha256"],
                                checkpoint_revision=policy.checkpoint["revision"],
                                checkpoint_source="published six-job frozen batch manifest"),
                comparison_scope="success-conditioned native sample; different scenes and task prompts"), inputs[0])


def descriptor(values):
    arr = np.asarray(values)
    return dict(samples=len(arr), minimum=arr.min(axis=0), maximum=arr.max(axis=0), median=np.median(arr,axis=0))


def joint_distribution(native, arx):
    return {q+1: dict(aloha=descriptor([v["observations"][q]["state"] for v in native.values()]),
                      arx=descriptor([v["state"][q]["state_archived"] for v in arx.values() if len(v["state"])>q]))
            for q in (0,1)}


def expert_reference(results, args, cfg):
    root = ROOT / "research/g2/native-expert-100"
    hashes = read(root / "evidence-hashes.json")
    arms = load_arms(args.assets,args.asset_audit,"aloha-agilex",cfg["calibration"]["aloha-agilex"])
    scenes = {}
    with (root / "expert_replay_100_results.csv").open() as stream:
        for row in DictReader(stream):
            if row["robot"] != "aloha-agilex":
                continue
            assert row["status"] == "SUCCESS"
            folder = root / "results" / Path(row["receipt_path"]).parent
            init = folder / "attempt" / "initial.json"
            digest = verified(init, root, hashes)
            evidence = read(init)
            block = matrix7(evidence["block_pose"])
            target = matrix7(evidence["target_pose"])
            trace_path = folder / "attempt" / "physics-trace.npz"
            trace_digest = verified(trace_path, root, hashes)
            with load_array(trace_path) as trace:
                names = [n.split(":")[-1] for n in trace["joint_names"].tolist()]
                values = trace["actual_qpos" if "actual_qpos" in trace.files else "actual_native"][0]
            joints = [values[[names.index(n) for n in arm.chain.names]] for arm in arms]
            root_pose = evidence["robot_base"] if "robot_base" in evidence else evidence["roots"][0]
            tools = fk(arms, [matrix7(root_pose)]*2, joints)
            scenes[int(row["seed"])] = dict(episode=int(row["episode"]), seed=int(row["seed"]),
                sha256=digest, trace_sha256=trace_digest, initial_path=str(init.relative_to(ROOT)), block_task=block,
                target_task=target, initial_tool_task=tools)
    assert len(scenes)==50
    matches = {}
    for g,x in results.items():
        seed = x["identity"]["scene_seed"]
        matched = scenes[seed]
        block, target = (np.asarray(x["initial"][key]) for key in ("block_task","placement_task"))
        def difference(a,b):
            error = pose_error(np.asarray(a),np.asarray(b))
            return dict(mm=1000*np.linalg.norm(error[:3]), deg=np.rad2deg(np.linalg.norm(error[3:])))
        matches[g] = dict(seed=seed, native_episode=matched["episode"],
                          block_difference=difference(block, matched["block_task"]),
                          target_difference=difference(target, matched["target_task"]),
                          virtual_home_tool_difference=[difference(a,b) for a,b in
                                      zip(x["virtual_initial_tcp_task"], matched["initial_tool_task"])])
    return dict(scope="50 published Aloha native expert CPU successes, seed-reconstructed static readbacks, "
                      "not pi0.5 training membership or expert actions used in policy",
                initial_block_xyz_task_mm=descriptor([xyz(x["block_task"]) for x in scenes.values()]),
                initial_target_xyz_task_mm=descriptor([xyz(x["target_task"]) for x in scenes.values()]),
                per_seed=scenes, matched_by_arx_trial=matches)


def sheet(images, path):
    w, h = 320, 240
    canvas = Image.new("RGB", (len(images)*w, (h+28)*3), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("Head", "Left wrist", "Right wrist")
    for col,(name, cameras) in enumerate(images.items()):
        for row,(key,label) in enumerate(zip(CAMERAS, labels)):
            y = row*(h+28)
            draw.text((col*w+8, y+7), f"{name} / {label}", fill="black")
            canvas.paste(Image.fromarray(cameras[key].transpose(1,2,0)), (col*w,y+28))
    canvas.save(path,optimize=True)


def main(args):
    args.output.mkdir(exist_ok=True)
    members = {entry["robot"]:entry for entry in read(args.asset_audit)["robots"]}
    calibration = read(ROOT / "research/g2/arx-frozen-input-audit/input_adapter_audit.json")["pinned_source_hashes"]
    with args.intrinsics.open("rb") as f:
        intrinsics_sha = file_digest(f, "sha256").hexdigest()
    assert intrinsics_sha == calibration["pace-_camera_config.yml"]["sha256"]
    fov = yaml.safe_load(args.intrinsics.read_text())["D435"]["fovy"]
    policy = policy_contract(args.image_tools)
    manifest = read(STUDY / "submission_receipts.json")["candidates"]
    results, audits, frames = {}, {}, {}
    for label, trial in manifest.items():
        results[label], audits[label], frames["ARX "+label] = arx_trial(label, trial, args, members, policy, fov)
    source_config = configs("G1", manifest["G1"]["episode"])
    native = {}
    for job in ALOHA:
        native[job], frames["Aloha "+job] = aloha_trial(job, args, members, source_config, policy, fov)
    ordered = {k:frames[k] for k in (*("Aloha "+j for j in ALOHA), *("ARX "+g for g in manifest))}
    sheet(ordered, args.output / "first_query_rgb.png")
    save(args.output / "first_query_analysis.json", dict(
        units="mm except homogeneous transform translations (m), angles from decoder in radians",
        convention="positive means block toward canonical TCP +Y; index vector [left,right]",
        geometry="mesh surface vs oriented block at the frozen query pose, no PhysX step",
        arx=results))
    save(args.output / "input_semantics_audit.json", dict(
        encoder="AlohaInputs(adapt_to_pi=False), quantile state/actions, absolute recovery by mask 6/-1/6/-1",
        provenance="real request and response archives; historic raw normalized tensor unavailable",
        arx=audits, intrinsics_sha256=intrinsics_sha, image_preprocessor_sha256=policy.image_source_sha256,
        aloha={k:[{n:v[n] for n in ("query","state","state_dtype","prompt","decode","camera")}
                  for v in run["observations"]] for k,run in native.items()},
        joint_distribution_by_query=joint_distribution(native,audits)))
    save(args.output / "aloha_arx_comparison.json", dict(
        aloha=native, expert_workspace=expert_reference(results, args, source_config), arx={k:{name:v[name] for name in
            ("identity","initial","virtual_initial_tcp_task","summary","physical","q2_bias_comparison","execution")}
                           for k,v in results.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("aloha", "arx", "assets", "asset_audit", "intrinsics", "image_tools", "output"):
        parser.add_argument(arg, type=Path)
    main(parser.parse_args())
