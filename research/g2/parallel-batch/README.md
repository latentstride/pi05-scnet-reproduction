# G2 parallel batch evidence

`configs/` freezes the seven experiments. `cpu/` contains actual CPU PhysX receipts with a fixed policy substitute, no GPU renderer and no checkpoint inference. `packages/batch.json` identifies the seven independently verified archives and their manifest, configuration and CPU receipt hashes. The project source lock identifies the upstream code and source evidence.

On the existing CPU PhysX host, with dependencies available and `CUDA_VISIBLE_DEVICES` empty:

```sh
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="scripts:$OPENPI_CLIENT_SRC:$SIM_DEPS"
r=research/g2/parallel-batch
for n in 1 2 3 4 5 6; do
  python -B "$r/cpu_admit.py" "$PACE_ROOT" "$FROZEN_RELEASE" \
    "$r/configs/G$n.json" "$FRESH_OUTPUT/G$n"
done
python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["reset"], indent=2))' \
  "$FRESH_OUTPUT/G2/cpu-admission.json" > "$FRESH_OUTPUT/G7-reference.json"
python -B "$r/cpu_admit.py" "$PACE_ROOT" "$FROZEN_RELEASE" \
  "$r/configs/G7.json" "$FRESH_OUTPUT/G7" --reference "$FRESH_OUTPUT/G7-reference.json"
python -B "$r/cpu_admit.py" "$PACE_ROOT" "$FROZEN_RELEASE" \
  "$r/configs/G5.json" "$FRESH_OUTPUT/G5-valley" --valley
```

Build deterministic packages from the historical standalone author archive without changing it:

```sh
python -B scripts/g2_parallel_package.py . \
  research/g2/standalone-review/g2-standalone-author.tar.gz \
  research/g2/parallel-batch "$FRESH_PACKAGE_DIRECTORY"
```

Extract per-trial raw closure, contact proxies, action counts and resource outcomes
from the saved files. Bilateral contact alone is insufficient to infer stable
holding or physical slip.

```sh
python -B "$r/summarize_trial.py" "$TRIAL_DIRECTORY"
```

Pair first exchanges after copying or linking the actual independent trial directories as `G1` to `G7`:

```sh
PYTHONPATH="scripts:$OPENPI_CLIENT_SRC" python -B "$r/summarize_pairs.py" "$TRIAL_DIRECTORIES" \
  --historical research/g2/job-127510335/result.json
```
