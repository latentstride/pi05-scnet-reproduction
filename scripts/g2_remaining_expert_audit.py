"""Audit all 854 original expert frame boundaries in archived real ARX physics.

Zero new physics. Restore actual q and block pose at each sampled frame; report
measured collisions separately from held-object predicted candidate geometry.
"""

import argparse, json
from pathlib import Path
from hashlib import sha256
from time import monotonic
import zipfile
import numpy as np
from g2_static_failure_replay import build, matrix
from g2_adapter import deadline
from g2_robotwin_backend import pose7
from robotwin_control_geometry import inverse_pose


def audit(a):
    a.output.mkdir(parents=True, exist_ok=False)
    report = {}
    check = deadline(150)
    with zipfile.ZipFile(a.selection) as z:
        for ep in range(3):
            started = monotonic()
            check()
            name = f"seed{min(ep, 1)}-B-arx-w1"
            cfg = json.loads((a.release / "configs" / f"{name}.json").read_text())
            initial = json.loads(z.read(name + "/initial-readback.json"))
            snapshot = json.loads(z.read(name + "/events-selected.json"))
            snap = next(x for x in snapshot if x.get("kind") == "snapshot")
            dpath = a.expert / f"episode{ep}" / "physics-trace.npz"
            d = np.load(dpath, allow_pickle=False)
            metadata = json.loads(dpath.with_name("initial.json").read_text())
            # Audit uses each archived expert's own bases and object poses.
            for side, b in zip(("left", "right"), metadata["robot_bases"]):
                snap["actual"][("left", "right").index(side)]["world_base"] = matrix(b)
            initial["objects_task"]["target"] = metadata["target_pose"]
            initial["objects_task"]["block"] = metadata["block_pose"]
            out = a.output / f"episode{ep}"
            out.mkdir()
            scene, arms, source, frames, bases, target, proxy, q, g = build(
                cfg,
                a.release / "asset-audit.json",
                a.assets,
                snap,
                initial,
                matrix(metadata["block_pose"]),
                out,
            )
            block = target.base.world.get_object(target.block_name)
            actor = next(x for x in scene.get_all_actors() if x.name == "box")
            import sapien

            rows = []
            for frame, tick in enumerate(d["frame_ticks"]):
                check()
                index = int(np.searchsorted(d["sample_ticks"], tick))
                if index == len(d["sample_ticks"]) or d["sample_ticks"][index] != tick:
                    raise ValueError("expert frame missing exact actual sample")
                state = d["actual_native"][index].reshape(2, 8)
                q, g = state[:, :6], state[:, 6:]
                pose = d["block_pose"][index]
                actor.set_pose(sapien.Pose(pose[:3], pose[3:]))
                target.refresh()
                # Holders are not invented: actual measured collision checking
                # uses actual block transform and original contact permissions.
                measured = target.measured_clear(q, g)
                measured_bad = target.last_collisions if not measured else []
                rows.append(
                    dict(
                        frame=frame,
                        physics_tick=int(tick),
                        actual_fcl_clear=bool(measured),
                        collisions=measured_bad,
                    )
                )
            rejected = [r for r in rows if not r["actual_fcl_clear"]]
            report[str(ep)] = dict(
                frames=len(rows),
                accepted=len(rows) - len(rejected),
                rejected=len(rejected),
                first_rejection=rejected[0] if rejected else None,
                all_rejections=rejected,
                source_trace_sha256=sha256(dpath.read_bytes()).hexdigest(),
                scene_physics_steps=0,
                gpu_used=False,
                wall_s=monotonic() - started,
                scope="exact archived expert frame boundaries: measured geometry only, no between-frame swept or physical replay claim",
            )
            (a.output / "expert-audit.json").write_text(
                json.dumps(report, default=lambda v: v.tolist(), indent=2) + "\n"
            )
            print("EXPERT", ep, len(rows), len(rejected), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("release", "selection", "assets", "expert", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    audit(p.parse_args())
