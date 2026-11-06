"""Validate every locked record before producing formal G2 summaries."""
import argparse
from hashlib import sha256
from pathlib import Path

from g2_episode import save_json
from summarize_experiments import check_formal_roster, decode, load_records


def collect(lock_path, lock_sha, results, output, *, data_kind="formal"):
    if data_kind not in ("formal", "diagnostic"):
        raise ValueError("expected a formal or diagnostic record kind")
    raw = lock_path.read_bytes()
    if sha256(raw).hexdigest() != lock_sha:
        raise ValueError('run lock hash mismatch')
    lock = decode(raw.decode())
    if lock.get('status') != 'frozen':
        raise ValueError('frozen run lock required')
    paths = []
    missing = []
    for trial in lock['episodes']:
        episode = trial['episode_id']
        if not episode or Path(episode).name != episode or episode in ('.', '..'):
            raise ValueError('invalid episode directory name')
        record = results / episode / f'{data_kind}-record.json'
        if not record.is_file():
            missing.append(episode)
        else:
            paths.append(record)
    if missing:
        save_json(output.with_name(f'missing-{data_kind}-records.json'), dict(missing_episode_ids=missing))
        raise ValueError(f'{len(missing)} locked episodes have no terminal {data_kind} record')
    # One object per episode directory. Validate through the existing list reader.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=output.parent) as tmp:
        assembled = Path(tmp) / 'records.json'
        decoded = [decode(p.read_text()) for p in paths]
        if any(row['episode_id'] != path.parent.name for path,row in zip(paths,decoded)):
            raise ValueError('formal record identity differs from its episode directory')
        save_json(assembled, decoded)
        rows = load_records([assembled], data_kind)
        check_formal_roster(lock_path, rows)
    save_json(output, rows)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('lock', type=Path)
    p.add_argument('lock_sha')
    p.add_argument('results', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    collect(a.lock, a.lock_sha, a.results, a.output)
