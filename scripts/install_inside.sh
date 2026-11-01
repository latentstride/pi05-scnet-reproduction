#!/usr/bin/env bash
set -euo pipefail
r=${PI05_ROOT:?Set PI05_ROOT to the project directory}
u=$r/tools/uv/bin/uv
p=$r/.venv/bin/python
s=$r/source/openpi-$(cat "$r/source/OPENPI_COMMIT")
"$u" pip install --python "$p" -r "$r/requirements.locked.txt"
"$u" pip install --python "$p" --no-deps -e "$r/source/lerobot" -e "$s/packages/openpi-client" -e "$s"
"$p" -c 'from openpi.training import config; from openpi.policies import policy_config; print(config.get_config("pi05_libero").name)'
"$u" pip freeze --python "$p" > "$r/logs/installed_packages.txt"
