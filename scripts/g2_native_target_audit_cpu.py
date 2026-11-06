"""Check captured Aloha arm states in the audited self-collision geometry on CPU."""

import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
from g2_assets import load_arms
from g2_collision import JointCollisionWorld
from g2_proxy_collision import collision_only_urdf


def run(a):
    output = a.output.resolve()
    for source in (a.release, a.assets, a.selection.parent):
        source = source.resolve()
        if output == source or source in output.parents:
            raise ValueError("analysis output must be outside the input directories")
    import sapien

    a.output.mkdir(parents=True, exist_ok=False)
    asset = json.loads((a.release / "asset-audit.json").read_text())
    (record,) = [r for r in asset["robots"] if r["robot"] == "aloha-agilex"]
    audit = json.loads(a.trial_audit.read_text())
    result = {}
    with zipfile.ZipFile(a.selection) as z:
        for name in sorted(n for n in audit if "A-native" in n):
            out = a.output / name
            out.mkdir()
            cfg = json.loads((a.release / "configs" / f"{name}.json").read_text())
            arms = load_arms(
                a.assets,
                a.release / "asset-audit.json",
                "aloha-agilex",
                cfg["calibration"]["aloha-agilex"],
            )
            scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            path = collision_only_urdf(
                a.assets / record["urdf_member"],
                record["urdf_sha256"],
                out / "aloha.urdf",
            )
            entity = loader.load(str(path))
            entity.set_name("native_aloha")
            checker = JointCollisionWorld.from_scene(
                scene,
                [entity, entity],
                [x.chain.names for x in arms],
                [x.grip.names for x in arms],
            )
            initial = json.loads(z.read(name + "/initial-readback.json"))
            samples = json.loads(z.read(name + "/control-samples.json"))
            first = audit[name]["first_tracking_error_over_50mm"]
            probes = [
                (
                    "initial",
                    np.array(
                        [initial["actual_arm_qpos_rad"][s] for s in ("left", "right")]
                    ),
                    samples[0]["actual_fingers"],
                )
            ]
            for label, event in [
                ("first_50mm", first["event"] if first else None),
                ("final", samples[-1]),
            ]:
                if event:
                    probes += [
                        (
                            label + "_actual",
                            event["actual_arm"],
                            event["actual_fingers"],
                        ),
                        (
                            label + "_target",
                            event["target_arm"],
                            event["actual_fingers"],
                        ),
                    ]
            checks = []
            for label, q, g in probes:
                clear = checker(np.asarray(q), np.asarray(g))
                checks.append(
                    dict(
                        label=label,
                        clear=bool(clear),
                        q=q,
                        fingers=g,
                        collisions=checker.last_collisions,
                    )
                )
            result[name] = dict(
                scope="articulation self-collision only; target arm q with recorded actual fingers; no task actors or dynamics",
                checks=checks,
                physics_steps=0,
                gpu_used=False,
            )
            print(name, [(r["label"], r["clear"]) for r in checks], flush=True)
            del scene, entity, checker
    (a.output / "native-self.json").write_text(
        json.dumps(result, default=lambda x: x.tolist(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("release", "assets", "selection", "trial-audit", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
