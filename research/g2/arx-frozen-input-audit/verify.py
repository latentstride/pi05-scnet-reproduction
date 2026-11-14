"""Cross-check the published audit with original array hashes and physics receipts."""
import csv
import gzip
import json
from io import BytesIO
from pathlib import Path

import numpy as np

from archive import digest, read


def verify(folder):
    evidence = folder / "evidence"
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        assert digest((evidence / filename).read_bytes()) == expected, filename
    audit = read(folder / "input_adapter_audit.json")
    assert audit["overall_observable_input_status"] == "INPUT_ADAPTER_VERIFIED"
    assert audit["full_historical_model_tensor_status"] == "INSUFFICIENT_EVIDENCE"
    for job, record in audit["jobs"].items():
        folder_job = evidence / job
        with gzip.open(folder_job / "request-plan-contact-excerpts.jsonl.gz", "rt") as stream:
            events = [json.loads(line) for line in stream]
        sent = [row for row in events if row["event"]["kind"] == "policy_request"]
        assert len(sent) == len(record["requests"])
        for query, archived in zip(record["requests"], sent, strict=True):
            assert query["event_line"] == archived["line"]
            assert query["tick"] == archived["event"]["stamp"]["tick"]
            assert query["prompt"] == archived["event"]["prompt"]
            assert query["state"] == archived["event"]["state"] == query["state_cpu_reconstructed"]
            raw = (folder_job / f"observation-{query['query']:05}.npz").read_bytes()
            assert digest(raw) == query["observation_npz_sha256"]
            with np.load(BytesIO(raw), allow_pickle=False) as views:
                assert set(views.files) == set(query["cameras"])
                for key, camera in query["cameras"].items():
                    assert digest(views[key].tobytes()) == camera["array_sha256"]
            raw = (folder_job / f"action-{query['query']:05}.npy").read_bytes()
            assert digest(raw) == query["action_np_sha256"]
            actions = np.load(BytesIO(raw), allow_pickle=False)
            assert actions.shape == (50,14)
            np.testing.assert_array_equal(actions[0], query["action_first_row"])
            np.testing.assert_array_equal(actions[-1], query["action_last_row"])
    with (folder / "policy_grasp_geometry.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert [(int(r["query"]),int(r["row"])) for r in table] == [
        (q,n) for q in (1,2) for n in range(1,51)]
    for q in (1,2):
        raw = (evidence/f"127999421/action-{q:05}.npy").read_bytes()
        archive = np.load(BytesIO(raw),allow_pickle=False)
        static = read(evidence/f"mesh-q{q}/receipt.json")
        assert static["source_action_sha256"] == digest(raw)
        for i,r in enumerate(table[(q-1)*50:q*50]):
            np.testing.assert_array_equal(json.loads(r["archive_raw_row"]),archive[i])
            assert r["action_sha256"] == static["source_action_sha256"]
            np.testing.assert_array_equal(json.loads(r["left_arx_depth_requested_mm"]),
                                           static["records"][i]["target_requested_finger_depth_mm"][0])
    assert table[35]["gpu_first_left_contact_tick"] == "5783"
    assert all(int(row["gpu_bilateral_left_steps"] or 0) == 0 for row in table)
    assert all(int(row["cpu_tail_bilateral_left_steps"] or 0) == 0 for row in table)
    assert next(r for r in table[50:] if float(r["left_raw_grip"]) <= .5)["row"] == "27"
    replay = read(evidence/"q2-physical/receipt.json")
    first_step, = [e for line in (evidence/"q2-physical/events.jsonl").read_text().splitlines()
                  if (e := json.loads(line)).get("physics_step_count")]
    assert replay["hold_no_advance"] and replay["fresh_request"]["validated"]
    assert replay["policy_source_sha256"] == table[50]["action_sha256"]
    assert replay["end_tick"] == first_step["tick"] == 5784
    assert len(first_step["contact_state"]["contact_fingers"]["left"]) == 1
    assert not replay["stable_grasp"]
    print("PASS: 17 requests, 100 geometry rows, complete array hashes, Q2 measured contact/reobservation")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
