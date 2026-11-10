"""Read Slurm status, print every query immediately, and stop at terminal state."""

import argparse
import json
import subprocess
import time

TERMINAL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def accounting(ids):
    reply = subprocess.run(
        [
            "sacct",
            "-nP",
            "-j",
            ",".join(ids),
            "-o",
            "JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = {}
    for line in reply.stdout.splitlines():
        row = line.split("|")
        if row[0] in ids:
            job, state, code, elapsed, start, end, node = row
            rows[job] = dict(
                state=state.split()[0].rstrip("+"),
                exit_code=code,
                elapsed=elapsed,
                start=start,
                end=end,
                node=node,
            )
    return rows


def monitor(
    ids, *, query=accounting, emit=print, pause=time.sleep, interval=30, queries=1
):
    known = {}
    failures = 0
    for _ in range(queries):
        try:
            known.update(query(ids))
            failures = 0
            report = dict(
                jobs={job: known.get(job, {"state": "UNKNOWN"}) for job in ids}
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            failures += 1
            report = dict(
                jobs=known, query_error=str(error), consecutive_errors=failures
            )
        emit(json.dumps(report))
        if failures >= 2:
            return 2
        if all(known.get(job, {}).get("state") in TERMINAL for job in ids):
            return 0
        if _ + 1 < queries:
            pause(interval)
    return 2 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ids", nargs="+")
    p.add_argument("--queries", type=int, default=1)
    a = p.parse_args()
    if (
        not all(x.isascii() and x.isdigit() and int(x) > 0 for x in a.ids)
        or a.queries < 1
    ):
        p.error("positive job IDs and query count required")
    return monitor(a.ids, queries=a.queries, emit=lambda data: print(data, flush=True))


if __name__ == "__main__":
    raise SystemExit(main())
