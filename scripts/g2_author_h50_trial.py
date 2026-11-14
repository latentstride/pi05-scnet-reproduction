"""Run the reviewed author H50 child only inside an approved unified allocation."""

import argparse
import os
from pathlib import Path
from shutil import disk_usage

from g2_author_h50_campaign import REFERENCE_FILE, REVIEW_FILE, TRIAL_MIN_FREE_BYTES
from g2_author_h50_limits import BUDGET, EPISODE, MODE, ResourceCensored, TrialClock
from g2_author_native_h50 import author_native_h50
from g2_final_deploy import authorization
from g2e_host_supervisor import interrupt_signals
from pace_g1_orchestrate import persist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "pace", "output", "model-startup"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--trial-start", type=float, required=True)
    parser.add_argument("--hard-deadline", type=float, required=True)
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or os.environ.get("SLURM_JOB_NAME") != "cpf-pace-g2":
        raise PermissionError("author H50 requires the unified allocation")
    approval = Path(os.environ["G2_FINAL_AUTHORIZATION"])
    auth = authorization(args.release, approval, job, starting=False)
    if (
        auth.get("job_id") != job
        or not (args.release / REVIEW_FILE).is_file()
        or approval.with_suffix(".used").read_text().strip() != job
    ):
        raise PermissionError(
            "author H50 overlay lacks active allocation authorization"
        )
    clock = TrialClock(BUDGET, args.trial_start, args.hard_deadline)
    try:
        clock.validate_entry()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if disk_usage(args.output.parent).free < TRIAL_MIN_FREE_BYTES:
            raise OSError("insufficient author H50 evidence disk headroom")
        with interrupt_signals():
            author_native_h50(
                args.pace,
                args.release,
                args.release / REFERENCE_FILE,
                args.output,
                EPISODE,
                args.uri,
                args.model_startup,
                clock=clock,
            )
    except ResourceCensored as error:
        # Before the diagnostic owned a scene, there is no physical receipt
        # that could authorize continuation. Preserve the known resource cause.
        args.output.mkdir(parents=True, exist_ok=False)
        persist(
            args.output / "receipt.json",
            dict(
                mode=MODE,
                episode_id=EPISODE,
                category="infrastructure_failure",
                outcome="RESOURCE_CENSORED",
                resource_reason=error.reason,
                success=False,
                fresh_reset_verified=False,
                cleanup=dict(scene_opened=False),
                trial_clock=clock.receipt(),
            ),
        )


if __name__ == "__main__":
    main()
