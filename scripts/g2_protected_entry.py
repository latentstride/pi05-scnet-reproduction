"""Explicit protected entry using the immutable historical author H50 child."""

import json
from pathlib import Path
import os
import sys


def main():
    # Keep the historical interpreter arguments and result schema intact.
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-receipt", type=Path, required=True)
    args, child = parser.parse_known_args()
    with args.mode_receipt.open("x") as stream:
        json.dump({"mode": "protected", "entry": "g2_author_h50_trial.py"}, stream)
    entry = Path(__file__).with_name("g2_author_h50_trial.py")
    os.execv(sys.executable, [sys.executable, str(entry), *child])


if __name__ == "__main__":
    main()
