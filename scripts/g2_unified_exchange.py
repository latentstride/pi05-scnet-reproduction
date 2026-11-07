"""Byte-level first-exchange comparison for the unified native controls."""

from hashlib import sha256
import json

import numpy as np

from pace_runtime import CAMERAS, validate_request, validate_response
from pace_g1_orchestrate import persist
from g2_initial import digest_json


def hash_array(array):
    value = np.ascontiguousarray(array)
    metadata = json.dumps(
        [value.dtype.str, value.shape], separators=(",", ":")
    ).encode()
    return sha256(metadata + b"\0" + value.tobytes()).hexdigest()


def exchange_receipt(request, response):
    validate_request(request)
    actions = validate_response(response)
    fields = dict(
        state=hash_array(request["state"]),
        cameras={name: hash_array(request["images"][name]) for name in CAMERAS},
        prompt=request["prompt"],
    )
    return dict(
        request_sha256=sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest(),
        chunk_sha256=hash_array(actions),
    )


class CheckedFirstExchange:
    """Reject divergent initial inference before the policy can return an action."""

    def __init__(self, policy, reference, output, initial):
        if reference.get("category") != "admitted" or not all(
            reference.get(key) for key in ("first_exchange", "initial_sha256")
        ):
            raise ValueError("native startup reference is incomplete")
        self.policy, self.reference, self.output, self.initial = (
            policy,
            reference,
            output,
            initial,
        )

    def infer_raw(self, request):
        response = self.policy.infer_raw(request)
        if self.reference is not None:
            observed = dict(
                first_exchange=exchange_receipt(request, response),
                initial_sha256=digest_json(json.loads(self.initial.read_bytes())),
            )
            expected = {key: self.reference[key] for key in observed}
            persist(self.output, dict(observed=observed, expected=expected))
            if observed != expected:
                raise ValueError("paired first request or chunk hash differs")
            self.reference = None
        return response

    infer = infer_raw
