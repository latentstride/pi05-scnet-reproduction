"""Cross-check generated evidence against the independently archived audit and invariants."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from input_semantics import ROOT

D = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def verify(directory):
    first = read(directory / "first_query_analysis.json")["arx"]
    semantics = read(directory / "input_semantics_audit.json")
    comparison = read(directory / "aloha_arx_comparison.json")
    with gzip.open(ROOT / "research/g2/arx-online-batch-20261115/root-cause/action-geometry.json.gz", "rt") as f:
        previous = json.load(f)
    counts = []
    for g, trial in first.items():
        assert trial["request"]["stamp"]["tick"] == 0
        assert trial["summary"]["first_requests_le"]["0.5"] is None
        assert trial["identity"]["checkpoint_file_matches"] == 28
        assert all(state["max_state_error"] == 0 for state in semantics["arx"][g]["state"])
        raw_goals = [x for x in previous[g]["rows"] if x["query"] == 1]
        assert len(trial["rows"]) == len(raw_goals) == 50
        for row, ref in zip(trial["rows"], raw_goals, strict=True):
            assert row["row"] == ref["row"]
            np.testing.assert_array_equal(row["raw_action"], ref["raw_action"])
            np.testing.assert_allclose(row["tool_geometry"]["block_xyz_tcp_mm"],
                                       ref["block_at_query_in_target_tcp_mm"], atol=1e-8, rtol=0)
        contact = trial["physical"]["events"].get("first_single")
        if g == "G4":
            assert contact is None
            assert not trial["physical"]["per_row_max_xy"]
            assert len(trial["execution"]["unexecuted_rejected_first_row"]) == 16
            continue
        assert contact["query"] == 1
        assert contact["row"] == trial["summary"]["first_single_finger_endpoint"]
        assert contact["block_xy_shift_mm"] < .03
        assert trial["physical"]["max_xy_strictly_before_first_contact_mm"] < .004
        q2 = trial["q2_bias_comparison"]
        assert q2["q1_signed_goal_y_mm"] > 24
        effect = q2["q2"]["effective_close"]
        np.testing.assert_allclose(effect["total_difference_mm"],
            effect["new_target_yaw_position_contribution_mm"] + effect["block_translation_contribution_mm"],
            rtol=0, atol=1e-10)
        assert abs(effect["block_translation_contribution_mm"]) < .005
        counts.append(contact["tick"])
    assert counts == [5767, 5784, 4855]

    for native in comparison["aloha"].values():
        assert native["closure"][0]["first_requests_le"]["0.5"] is None
        assert native["contacts"]["first_bilateral"]
    matches = comparison["expert_workspace"]["matched_by_arx_trial"]
    assert len(comparison["expert_workspace"]["per_seed"]) == 50
    assert all(m["block_difference"]["mm"] < .002 for m in matches.values())
    wrist = semantics["arx"]["G3"]["cameras"][1]["cam_left_wrist"]
    assert wrist["independently_clamped_max_error_px"] > 31
    assert wrist["visible_red_bounds_max_error_px"] < .3
    print("PASS: 200 raw ARX rows, 3 pre-contact offsets, 16 G4 rejections, "
          "3 Aloha successes, 50 expert resets, and archived camera clipping regression")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=D)
    verify(parser.parse_args().directory)
