"""Control-flow regression independent of model and simulator resources."""

from dataclasses import dataclass
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from g2_adapter import G2Error, Stamp
from g2_arx_direct import execute_prefix
from robotwin_control_geometry import Status


@dataclass
class Plan:
    waypoint_ticks: tuple = (1,)


class DirectPrefixTest(unittest.TestCase):
    def run_prefix(self, *, contact_row=None, blocked_row=None):
        backend = NS(tick=0, dt=0.004, arms=(), success=lambda: False)
        tracker = NS(
            snapshot=lambda: dict(
                holders=[],
                releasing=[],
                contact_fingers={
                    "left": [1] if backend.tick == contact_row else [],
                    "right": [],
                },
            )
        )
        world = NS(tracker=tracker, refresh=Mock(), measured_clear=lambda *_: True)
        backend.collision = world
        backend.take_replan = lambda: None
        backend.capture_state = lambda: NS(
            stamp=Stamp("test", backend.tick, backend.tick * backend.dt)
        )
        starts = []
        completed = []

        def execute(plan, log, remaining, *, stop_on_contact):
            assert stop_on_contact
            backend.tick += 1
            return 1

        backend.execute = execute

        def plan_row(row, snap, remaining):
            starts.append(snap.stamp.tick)
            if row == blocked_row:
                return NS(
                    ok=False,
                    status=Status.COLLISION,
                    detail="fixture rejection",
                    diagnostic={},
                )
            return NS(ok=True, payload=Plan(), diagnostic={})

        with patch(
            "g2_arx_direct.read_snapshot", return_value=(None, None, None, None)
        ):
            try:
                receipt = execute_prefix(
                    backend,
                    backend.capture_state(),
                    range(8),
                    plan_row,
                    lambda: 1.0,
                    lambda _: None,
                    lambda: completed.append(backend.tick),
                )
            except G2Error as error:
                receipt = error
        return receipt, starts, completed

    def test_eight_rows_replan_from_consecutive_measurements(self):
        receipt, starts, completed = self.run_prefix()
        self.assertEqual(starts, list(range(8)))
        self.assertEqual(completed, list(range(1, 9)))
        self.assertEqual(receipt.completed, 8)

    def test_first_measured_contact_retires_tail(self):
        receipt, starts, completed = self.run_prefix(contact_row=5)
        self.assertEqual(len(starts), len(completed))
        self.assertEqual(receipt.completed, 5)
        self.assertEqual(receipt.stamp.tick, 5)
        self.assertEqual(receipt.reason, "contact_requires_observation")

    def test_rejected_row_executes_neither_itself_nor_any_tail(self):
        error, starts, completed = self.run_prefix(blocked_row=5)
        self.assertEqual(starts, list(range(6)))
        self.assertEqual(completed, list(range(1, 6)))
        self.assertEqual(error.status, Status.COLLISION)


class DirectExchangeTest(unittest.TestCase):
    def test_contact_requeries_at_measured_stop_and_uses_only_new_chunk(self):
        import numpy as np
        from g2_arx_direct import DirectController, Prefix

        ticks = iter((0, 7))

        def capture():
            tick = next(ticks)
            return (
                NS(
                    stamp=Stamp("test", tick, tick * 0.004),
                    objects_task={"block": tick, "target": 1},
                ),
                None,
            )

        request_ticks = []

        def observe(snapshot, prompt, remaining):
            request_ticks.append(snapshot.objects_task["block"])
            return NS(ok=True, payload={"state": snapshot.stamp.tick})

        def infer(request):
            return {"actions": np.full((50, 14), request["state"], dtype=float)}

        def run_prefix(backend, snapshot, rows, adapter, remaining, log, consumed):
            call_rows.append(rows.copy())
            consumed()
            tick = snapshot.stamp.tick + 7
            return Prefix(
                1, Stamp("test", tick, tick * 0.004), "measured_first_contact"
            )

        call_rows = []
        controller = DirectController(
            NS(observe=observe, policy_row=Mock()), NS(capture=capture)
        )
        contract = NS(
            validate_request=lambda request: None,
            validate_response=lambda response: response["actions"],
        )
        with patch.dict("sys.modules", {"pace_runtime": contract}):
            with patch("g2_arx_direct.execute_prefix", side_effect=run_prefix):
                for _ in range(2):
                    controller.cycle(
                        NS(infer=infer), "test", 24, lambda: 1.0, lambda _: None
                    )
        self.assertEqual(request_ticks, [0, 7])
        self.assertEqual(controller.policy_actions, 2)
        np.testing.assert_array_equal(call_rows[1], np.full((24, 14), 7.0))
