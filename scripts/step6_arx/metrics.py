"""Read-only stage measures. These thresholds never enter collision or control code."""

import math

METRICS = (
    "left_approach", "left_contact", "left_bilateral", "left_grasp", "left_lift",
    "sideways_push", "unexpected_collision", "ik_rejection", "collision_rejection",
    "right_approach", "right_contact", "right_bilateral", "shared_hold", "right_takeover",
    "drop", "target_region", "released_at_target", "native_success",
)
STAGES = {"left_approach": "left_approach", "left_grasp_and_lift": "left_lift",
          "handover": "right_takeover", "placement": "native_success"}
NEAR_CENTER_M = 0.12
LIFT_M = 0.03
PUSH_XY_M = 0.02
PUSH_Z_M = 0.01
TAKEOVER_S = 0.04
DROP_M = 0.03


class Stages:
    def __init__(self):
        self.first = {}
        self.observable = set()
        self.initial = None
        self.clear_left_since = None
        self.pre_lift_touch = False
        self.last = None
        self.last_holders = None
        self.lost_at_z = None
        self.min_distance = {"left": None, "right": None}
        self.peak_lift = None

    def record(self, name, passed, event):
        self.observable.add(name)
        if passed and name not in self.first:
            self.first[name] = {key: event[key] for key in ("tick", "sim_s", "action_index", "query_index", "unix_ns")
                                if key in event}

    def step(self, row):
        if self.last and self.last["tick"] == row["tick"]:
            return
        if self.last and self.last["tick"] > row["tick"]:
            raise ValueError("out of order physics samples")
        self.last = row
        block = row.get("block_pose_world")
        tcp = row.get("tcp_world")
        state = row.get("contact_state")
        if block is None:
            return
        if self.initial is None:
            self.initial = block[:3]
        z = block[2]
        dz = z - self.initial[2]
        self.peak_lift = max(self.peak_lift or 0., dz)
        if tcp is not None:
            for i, side in enumerate(("left", "right")):
                d = math.dist([r[3] for r in tcp[i][:3]], block[:3])
                self.min_distance[side] = min(d, self.min_distance[side] if self.min_distance[side] is not None else d)
                self.record(side + "_approach", d <= NEAR_CENTER_M, row)
        functional = row.get("box_functional_point_world")
        target = row.get("target_functional_point_world")
        if functional is not None and target is not None:
            in_region = all(abs(a - b) < e for a, b, e in zip(functional, target, (0.03, 0.03, 0.01)))
            self.record("target_region", in_region, row)
        else:
            in_region = None
        if row.get("native_success") is not None:
            self.record("native_success", row["native_success"], row)
        if state is None:
            return
        holders = set(state["holders"])
        releasing = set(state["releasing"])
        fingers = state["contact_fingers"]
        for side in ("left", "right"):
            self.record(side + "_contact", bool(fingers[side]), row)
            self.record(side + "_bilateral", len(fingers[side]) == 2, row)
        self.pre_lift_touch |= bool(fingers["left"])
        self.record("left_grasp", "left" in holders, row)
        self.record("left_lift", "left" in holders and dz >= LIFT_M, row)
        push = (self.pre_lift_touch and "left_lift" not in self.first and not holders and abs(dz) <= PUSH_Z_M
                and math.dist(block[:2], self.initial[:2]) >= PUSH_XY_M)
        self.record("sideways_push", push, row)
        self.record("shared_hold", holders == {"left", "right"} and not releasing, row)
        aperture = row.get("actual_gripper_aperture")
        if aperture is not None:
            takeover_candidate = ("shared_hold" in self.first and "left_lift" in self.first
                                  and holders == {"right"} and not releasing and not fingers["left"]
                                  and aperture[0] > 0.8)
            if takeover_candidate:
                if self.clear_left_since is None:
                    self.clear_left_since = row["sim_s"]
            else:
                self.clear_left_since = None
            self.record("right_takeover", self.clear_left_since is not None and
                        row["sim_s"] - self.clear_left_since + 1e-9 >= TAKEOVER_S, row)
            if in_region is not None:
                self.record("released_at_target", in_region and not holders and not any(fingers.values())
                            and aperture[1] > 0.8 and "right_takeover" in self.first, row)
        if self.last_holders and not holders and "left_lift" in self.first and in_region is False:
            self.lost_at_z = z
        if holders or in_region:
            self.lost_at_z = None
        if in_region is not None:
            self.record("drop", self.lost_at_z is not None and z <= self.lost_at_z - DROP_M, row)
        self.last_holders = holders
        if row.get("contacts") is not None:
            def unexpected(contact):
                roles = [r for r in contact["roles"] if r is not None]
                if not (contact["active"] and roles):
                    return False
                # Only visible block contact from the calibrated finger links is expected.
                # Intra-arm contacts use the existing FCL audit instead of redefining ignores.
                if len(roles) == 2 and roles[0][0] == roles[1][0]:
                    return False
                return not (contact["involves_block"] and len(roles) == 1
                            and roles[0] in row.get("allowed_block_contact_roles", []))
            self.record("unexpected_collision", any(unexpected(c) for c in row["contacts"]), row)

    def safety(self, event):
        status = event["status"]
        if status == "collision":
            self.record("collision_rejection", True, event)
            if event.get("diagnostic", {}).get("geometry") == "actual":
                self.record("unexpected_collision", True, event)

    def summary(self):
        result = {name: {"status": ("observed" if name in self.first else
                                    "not_observed" if name in self.observable else "unavailable"),
                         "first": self.first.get(name)} for name in METRICS}
        return dict(metrics=result, stages={name: result[metric] for name, metric in STAGES.items()},
                    min_tcp_to_center_m=self.min_distance, peak_block_lift_m=self.peak_lift)
