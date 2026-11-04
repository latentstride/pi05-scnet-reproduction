"""Persist phase timing and watchdog stacks independently of the rollout summary."""
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import json
from time import monotonic
import traceback


class Diagnostics:
    def __init__(self, events, stacks, summary, stall_seconds):
        self.events = events
        self.stacks = stacks
        self.summary = summary
        self.stall_seconds = stall_seconds

    def record(self, stage, event, **details):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.events.write(json.dumps({"stage": stage, "event": event, "utc": timestamp, **details}) + "\n")
        self.events.flush()
        print(f"[{timestamp}] {event} {stage}", flush=True)

    @contextmanager
    def phase(self, stage):
        start = monotonic()
        self.record(stage, "start")
        self.stacks.write(f"\n[{stage}] watchdog every {self.stall_seconds:g}s\n")
        self.stacks.flush()
        faulthandler.dump_traceback_later(self.stall_seconds, repeat=True, file=self.stacks)
        try:
            yield
        except BaseException as exc:
            self.summary.setdefault("failed_phase", stage)
            traceback.print_exc(file=self.stacks)
            self.stacks.flush()
            self.record(stage, "error", elapsed_s=monotonic() - start,
                        error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.record(stage, "end", elapsed_s=monotonic() - start)
        finally:
            faulthandler.cancel_dump_traceback_later()
