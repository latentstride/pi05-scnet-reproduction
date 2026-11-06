"""Run fixed commands sequentially as members of the supervisor's process group."""
import argparse
import json
from pathlib import Path
import subprocess


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('commands',type=Path)
    p.add_argument('--monitor',type=Path)
    a=p.parse_args()
    for argv in json.loads(a.commands.read_text()):
        subprocess.run(argv,check=True)
    if a.monitor is not None:
        samples=[]
        for line in a.monitor.read_text().splitlines():
            fields=line.split(',')
            if len(fields)==4 and all(x.strip().isdigit() for x in fields[2:]):
                samples.append(tuple(map(int,fields[2:])))
        if not samples or min(total-used for used,total in samples)<1024:
            raise SystemExit('startup GPU headroom below 1024 MiB or missing measurement')
