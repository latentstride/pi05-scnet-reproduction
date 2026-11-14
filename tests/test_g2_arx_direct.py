"""Control-flow regression independent of model and simulator resources."""

from dataclasses import dataclass
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from g2_adapter import G2Error, Stamp
from g2_arx_direct import execute_prefix,UnexecutedContactProposal
from robotwin_control_geometry import Status


@dataclass
class Plan:
    waypoint_ticks: tuple = (1,)


class ContactProposalTest(unittest.TestCase):
    def test_only_known_movable_finger_pair_is_deferred_in_unexecuted_geometry(self):
        world=NS(profile_id='real',block_name='block',allowed_pairs={('arx_left','finger')},
                 allows=lambda row:False)
        future=UnexecutedContactProposal(world)
        row=dict(object_name1='block',link_name1='block',object_name2='arx_left',
                 link_name2='finger',max_penetration_m=.03)
        self.assertTrue(future.allows(row))
        self.assertFalse(world.allows(row))
        self.assertFalse(future.allows(dict(row,link_name2='wrist')))
        self.assertFalse(future.allows(dict(row,object_name1='table')))
        self.assertFalse(future.allows(dict(row,max_penetration_m=None)))
        self.assertNotEqual(future.profile_id,world.profile_id)


class DirectPrefixTest(unittest.TestCase):
    def run_prefix(self, *, contact_row=None, blocked_row=None, rows=8):
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
                    range(rows),
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

    def test_full_prediction_is_checked_row_by_row_until_contact(self):
        receipt, starts, completed = self.run_prefix(contact_row=47, rows=50)
        self.assertEqual(starts, list(range(47)))
        self.assertEqual(completed, list(range(1, 48)))
        self.assertEqual(receipt.reason, "contact_requires_observation")

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

    def contact_context(self, *, blocked=None, transition=None, projected=True,
                        other_arm=None, release=None):
        """Exercise real prefix control with slow physical operations replaced."""
        import json
        from pathlib import Path
        import numpy as np

        source = Path(__file__).resolve().parents[1] / 'research/g2/arx-systematic-20261114/inputs/127999421-q2.json'
        rows = np.asarray(json.loads(source.read_text())['chunk'])
        backend = NS(tick=0, dt=.004, arms=(), success=lambda: False)
        def state():
            return dict(holders=['left'] if transition == backend.tick else [],
                        releasing=['left'] if release == backend.tick else [],
                        contact_fingers={'left':[11], 'right':[22] if other_arm==backend.tick else []})
        backend.collision = NS(tracker=NS(snapshot=state),refresh=lambda:None,
                               measured_clear=lambda *_:True)
        backend.capture_state=lambda:NS(stamp=Stamp('test',backend.tick,backend.tick*.004))
        backend.take_replan=lambda:None
        executed=[]
        def execute(plan,log,remaining,*,stop_on_contact):
            executed.append((backend.tick,float(rows[backend.tick,6]),stop_on_contact))
            backend.tick+=1
            return 1
        backend.execute=execute
        def plan(row,snapshot,remaining):
            if backend.tick == blocked:
                return NS(ok=False,status=Status.COLLISION,detail='ARX finger/block',diagnostic={})
            return NS(ok=True,payload=Plan(),diagnostic={'contact_barrier':True} if projected else {})
        with patch('g2_arx_direct.read_snapshot',return_value=(None,None,None,None)):
            try:
                receipt=execute_prefix(backend,backend.capture_state(),rows[:32],plan,
                    lambda:10.,lambda _:None,lambda:None)
            except G2Error as error:
                receipt=error
        return receipt,executed

    def test_observed_contact_does_not_discard_safe_original_closure_tail(self):
        receipt,executed=self.contact_context()
        self.assertEqual(receipt.completed,32)
        self.assertLess(executed[-1][1],.1)
        self.assertTrue(all(not row[2] for row in executed))

    def test_contact_context_keeps_real_arx_rejection_before_closure(self):
        error,executed=self.contact_context(blocked=7)
        self.assertEqual(len(executed),7)
        self.assertIsInstance(error,G2Error)
        self.assertEqual(error.status,Status.COLLISION)

    def test_confirmed_grasp_transition_still_requires_fresh_observation(self):
        receipt,executed=self.contact_context(transition=5)
        self.assertEqual(receipt.completed,5)
        self.assertEqual(len(executed),5)

    def test_new_receiver_contact_still_retires_tail(self):
        receipt,executed=self.contact_context(other_arm=5)
        self.assertEqual(receipt.completed,5)
        self.assertEqual(receipt.reason,'contact_requires_observation')

    def test_release_transition_still_retires_tail(self):
        receipt,executed=self.contact_context(release=5)
        self.assertEqual(receipt.completed,5)
        self.assertEqual(receipt.reason,'measured_contact_transition')

    def test_safe_subsegments_keep_original_row_and_count_it_once(self):
        backend=NS(tick=0,dt=.004,arms=(),success=lambda:False)
        state=dict(holders=[],releasing=[],contact_fingers={'left':[11],'right':[]})
        backend.collision=NS(tracker=NS(snapshot=lambda:state),refresh=lambda:None,
                             measured_clear=lambda *_:True)
        backend.capture_state=lambda:NS(stamp=Stamp('test',backend.tick,backend.tick*.004))
        backend.take_replan=lambda:None
        seen=[];consumed=[]
        def plan(row,snapshot,remaining):
            seen.append((row,snapshot.stamp.tick))
            return NS(ok=True,payload=Plan(),diagnostic=dict(
                partial_policy_row=len(seen)<3,row_duration_limit_s=1.))
        def execute(plan,log,remaining,**kwargs):backend.tick+=1;return 1
        backend.execute=execute
        with patch('g2_arx_direct.read_snapshot',return_value=(None,None,None,None)):
            result=execute_prefix(backend,backend.capture_state(),[10,20],plan,
                lambda:10.,lambda _:None,lambda:consumed.append(backend.tick))
        self.assertEqual(seen,[(10,0),(10,1),(10,2),(20,3)])
        self.assertEqual(consumed,[1,4])
        self.assertEqual(result.completed,2)

    def test_partial_row_cannot_reset_existing_motion_duration_limit(self):
        backend=NS(tick=0,dt=.004,arms=(),success=lambda:False)
        state=dict(holders=[],releasing=[],contact_fingers={'left':[11],'right':[]})
        backend.collision=NS(tracker=NS(snapshot=lambda:state),refresh=lambda:None,
                             measured_clear=lambda *_:True)
        backend.capture_state=lambda:NS(stamp=Stamp('test',backend.tick,backend.tick*.004))
        backend.take_replan=lambda:None
        def execute(plan,log,remaining,**kwargs):backend.tick+=1;return 1
        backend.execute=execute
        plan=lambda *_:NS(ok=True,payload=Plan(),diagnostic=dict(
            partial_policy_row=True,row_duration_limit_s=.008))
        with patch('g2_arx_direct.read_snapshot',return_value=(None,None,None,None)):
            with self.assertRaisesRegex(G2Error,'motion duration'):
                execute_prefix(backend,backend.capture_state(),[10],plan,
                    lambda:10.,lambda _:None,lambda:None)
        self.assertEqual(backend.tick,2)


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
