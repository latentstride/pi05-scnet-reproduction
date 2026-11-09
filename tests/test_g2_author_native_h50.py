"""CPU contract and shared-service transport tests for the isolated entry."""

from copy import deepcopy
from threading import Thread
import unittest
from unittest.mock import Mock

import numpy as np
from websockets.sync.server import serve

from g2_adapter import G2Error
from g2_author_native_h50 import MODE, contact_readback, may_continue
from g2_author_native_h50_service import handler, shared_connection
from g2_candidate_stop import failure_sha256
from g2e_model_entry import reset_remote
from pace_runtime import CAMERAS, open_policy


class TestRng:
    """Deterministic key stand-in; tests transport and chaining, not JAX."""

    @staticmethod
    def key(seed):
        return (seed, 0)

    @staticmethod
    def key_impl(_):
        return "test_key"

    @staticmethod
    def key_data(key):
        return np.array(key, dtype=np.uint32)

    @staticmethod
    def split(key):
        return (key[0], key[1] + 1), (key[0], key[1] + 101)


class Policy:
    _is_pytorch_model = False
    _rng = (99, 0)

    def infer(self, request):
        self._rng, _ = TestRng.split(self._rng)
        return dict(actions=np.tile(request["state"], (50, 1)))


class AuthorH50Contract(unittest.TestCase):
    def test_measured_empty_contact_is_distinct_from_unavailable(self):
        self.assertEqual(contact_readback(Mock(get_contacts=lambda: [])), ([], []))
        with self.assertRaises(G2Error):
            contact_readback(Mock(get_contacts=lambda: None))
        with self.assertRaises(RuntimeError):
            contact_readback(
                Mock(get_contacts=Mock(side_effect=RuntimeError("engine")))
            )

    def test_author_service_request_identity_rng_chain_and_legacy_connection(self):
        policy = Policy()
        dispatch = handler(policy, "checkpoint", jax_random=TestRng)
        request = dict(
            state=np.arange(14, dtype=np.float64),
            images={k: np.zeros((3, 240, 320), np.uint8) for k in CAMERAS},
            prompt="test",
        )
        with serve(dispatch, "127.0.0.1", 0) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
            try:
                with shared_connection(uri, "a", lambda: 2.0)() as remote:
                    self.assertEqual(remote.reset_receipt["key"]["words"], [0, 0])
                    first = remote.infer(request)["author_rng"]
                    request["state"] = request["state"].copy()
                    request["state"][0] += 1
                    second = remote.infer(request)["author_rng"]
                    self.assertEqual((first["query"], second["query"]), (0, 1))
                    self.assertEqual(first["after"], second["before"])
                    self.assertEqual(second["sample"]["words"], [0, 102])
                    self.assertNotEqual(
                        first["request_sha256"], second["request_sha256"]
                    )
                # Legacy runs still receive their original envelope and no
                # author-only keys. Inject the key factory since this is CPU-only.
                from unittest.mock import patch
                from g2e_model_entry import reset_rng

                with patch(
                    "g2e_model_entry.reset_rng",
                    side_effect=lambda p, seed: reset_rng(p, seed, TestRng.key),
                ):
                    with open_policy(uri, "checkpoint", 2.0) as remote:
                        reset_remote(remote, 123, "legacy")
                        self.assertNotIn("author_rng", remote.infer(request))
                        self.assertEqual(policy._rng, (123, 1))
            finally:
                server.shutdown()
                thread.join(3)

    def test_no_continue_without_same_reset_proof_clean_process_and_live_service(self):
        result = dict(
            failure=dict(code="collision", diagnostic=dict(geometry="candidate"))
        )
        proof = dict(
            kind="candidate_rejection",
            episode="first",
            tick=0,
            state_sha256="a" * 64,
            failure_sha256=failure_sha256(result),
            contact_sha256="b" * 64,
            contact_manifold_count=0,
            contact_source="sapien.Scene.get_contacts",
            contact_sample_measured=True,
            **dict.fromkeys(
                (
                    "terminal_checks_passed",
                    "hold_verified",
                    "hold_complete",
                    "close_complete",
                    "scene_context_closed",
                    "no_physics_advance",
                ),
                True,
            ),
        )
        receipt = dict(
            mode=MODE,
            episode_id="first",
            category="controlled_candidate_stop",
            cleanup=dict.fromkeys(
                (
                    "scene_opened",
                    "hold_complete",
                    "close_complete",
                    "scene_context_closed",
                ),
                True,
            ),
            result=result,
            success=False,
            fresh_reset_verified=True,
            candidate_stop=proof,
            initial_measurement=dict(
                status="measured", state_sha256="a" * 64, contact_sha256="b" * 64
            ),
        )

        def admitted(row):
            return may_continue(
                row, "first", process_group_reaped=True, shared_service_alive=True
            )

        self.assertTrue(admitted(receipt))
        self.assertFalse(
            may_continue(
                receipt, "next", process_group_reaped=True, shared_service_alive=True
            )
        )
        self.assertFalse(
            may_continue(
                receipt, "first", process_group_reaped=False, shared_service_alive=True
            )
        )
        self.assertFalse(
            may_continue(
                receipt, "first", process_group_reaped=True, shared_service_alive=False
            )
        )
        for update in (
            {"contact_sha256": "c" * 64},
            {"contact_sample_measured": False},
            {"hold_verified": False},
            {"failure_sha256": "d" * 64},
        ):
            invalid = deepcopy(receipt)
            invalid["candidate_stop"].update(update)
            self.assertFalse(admitted(invalid))
        invalid = deepcopy(receipt)
        invalid["result"]["failure"]["diagnostic"]["nested"] = {"geometry": "actual"}
        self.assertFalse(admitted(invalid))
        for category in ("actual_safety_stop", "infrastructure_failure"):
            self.assertFalse(admitted(dict(receipt, category=category)))
