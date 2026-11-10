"""Bind one frozen G1–G7 condition to the existing standalone supervisor."""

from dataclasses import replace

from g2_author_h50_limits import BUDGET
from g2_author_h50_receipt import may_continue
from g2_final_deploy import authorization, read, verify
from g2_parallel_spec import digest, load

CONFIG = "parallel-config.json"
MIN_DISK_GIB = 8


def verify_package(release):
    plan = verify(release)
    spec = load(release / CONFIG)
    if (
        read(release / "parallel-admission.json")
        != dict(submission_allowed=False, automatic_retry=False, spec=spec)
        or plan["checkpoint"]["revision"] != spec["checkpoint_revision"]
        or plan["checkpoint"]["norm_stats_sha256"] != spec["norm_stats_sha256"]
        or (release / "scripts/g2_parallel_deploy.py").exists()
        or 'cd "${G2_PROJECT_ROOT:'
        not in (release / "scripts/g2_graphics.sh").read_text()
    ):
        raise ValueError("standalone matrix package identity differs")
    r = spec["resources"]
    selected = dict(
        mode="matrix",
        episode=spec["episode_id"],
        config=spec,
        wall_s=r["wall_s"],
        startup_s=r["model_start_s"],
        hard_s=r["trial_hard_s"],
        port_base=spec["port_base"],
        min_disk_gib=MIN_DISK_GIB,
        fee_cap="7.20",
    )
    return plan, selected


def bound_authorization(release, path, job, spec, *, starting=True):
    auth = authorization(release, path, job, starting=starting)
    expected = dict(
        job_id=job,
        experiment=spec["id"],
        episode_id=spec["episode_id"],
        standalone_mode="matrix",
        allocation_s=spec["resources"]["wall_s"],
        configuration_sha256=digest(release / CONFIG),
        resources=spec["resources"],
        allocations_authorized=1,
        automatic_retry=False,
    )
    if (
        not job.isascii()
        or not job.isdigit()
        or not int(job)
        or any(auth.get(k) != v for k, v in expected.items())
    ):
        raise PermissionError("authorization does not bind this standalone matrix job")
    return auth


def trial_runtime(plan, runtime, release, output, selected, job, port):
    """Only change the model URI and single child/receipt binding."""
    spec = selected["config"]
    episode, r = spec["episode_id"], spec["resources"]
    budget = replace(
        BUDGET,
        soft_s=r["trial_soft_s"],
        hard_s=r["trial_hard_s"],
        request_s=r["request_s"],
    )
    template = next(iter(runtime["commands"].values()))
    sim = template[: template.index(str(release / "scripts/g2_final_trial.py"))]
    pace = template[template.index("--pace") + 1]
    approval = next(
        a.split("=", 1)[1] for a in sim if a.startswith("G2_FINAL_AUTHORIZATION=")
    )
    target = output / "trials" / episode
    target.parent.mkdir()
    uri = f"ws://127.0.0.1:{port}"
    child = sim + [
        str(release / "scripts/g2_parallel_worker.py"),
        "--release",
        str(release),
        "--output",
        str(target),
        "--pace",
        pace,
        "--uri",
        uri,
        "--model-startup",
        str(output / "model/startup.json"),
        "--authorization",
        approval,
    ]
    model = runtime["model"].copy()
    model[model.index(str(release / "scripts/g2e_model_entry.py"))] = str(
        release / "scripts/g2_author_native_h50_service.py"
    )
    model[model.index("--port") + 1] = str(port)
    model.insert(1, f"G2_STANDALONE_CACHE={output / 'cache'}")
    runtime.pop("validate_startup", None)
    runtime.update(
        model=model,
        ready_marker=f"checkpoint policy on {uri}",
        startup_cap_s=r["model_start_s"],
        probes={},
        commands={episode: child},
        read_result=lambda _: read(target / "receipt.json"),
        standalone_trial=dict(
            episode=episode,
            budget=budget,
            accept=lambda receipt, clock, alive: accepted_receipt(
                receipt, spec, clock, alive
            ),
        ),
    )
    row = dict(
        episode_id=episode,
        embodiment="aloha-agilex",
        seed=spec["scene_seed"],
        method=spec["method"],
        hard_timeout_s=budget.hard_s,
    )
    return dict(plan, trials=[row]), runtime


def accepted_receipt(receipt, spec, clock, alive):
    if spec["mode"] == "protected":
        return may_continue(
            receipt,
            spec["episode_id"],
            process_group_reaped=True,
            shared_service_alive=alive,
            clock=clock,
        )
    if (
        not alive
        or receipt.get("mode") != spec["mode"]
        or receipt.get("method") != spec["method"]
        or receipt.get("episode_id") != spec["episode_id"]
        or receipt.get("scene_closed") is not True
        or receipt.get("trial_clock") != clock.receipt()
    ):
        return False
    count = receipt.get("completed_actions")
    if type(count) is not int or not 0 <= count <= spec["action_cap"]:
        return False
    outcome = receipt.get("outcome")
    return (
        (outcome == "SUCCESS" and receipt.get("success") is True and count > 0)
        or (
            outcome == "TASK_INCOMPLETE"
            and receipt.get("success") is False
            and count == spec["action_cap"]
        )
        or (
            outcome == "RESOURCE_CENSORED"
            and receipt.get("success") is False
            and not receipt.get("request_in_flight")
        )
    )
