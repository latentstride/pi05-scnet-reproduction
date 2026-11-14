"""Capture failures before the parallel Python deployment can enter its logger."""

import json
import runpy
import sys
import traceback


if __name__ == "__main__":
    target = sys.argv.pop(1)
    sys.argv[0] = target
    print(
        json.dumps(
            dict(
                stage="python_bootstrap", state="passed", python=sys.version.split()[0]
            )
        ),
        flush=True,
    )
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as error:
        if error.code:
            print(
                json.dumps(
                    dict(
                        stage="python_exit",
                        outcome="INFRASTRUCTURE_FAILURE",
                        exit_code=error.code,
                    ),
                    default=str,
                ),
                file=sys.stderr,
                flush=True,
            )
        raise
    except (Exception, KeyboardInterrupt) as error:
        print(
            json.dumps(
                dict(
                    stage="python_bootstrap",
                    outcome="INFRASTRUCTURE_FAILURE",
                    exception=type(error).__name__,
                    detail=str(error),
                    exit_code=1,
                )
            ),
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise SystemExit(1)
