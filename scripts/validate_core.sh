#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_container.sh bash -c '
set -e
r=$PI05_ROOT
$r/tools/uv/bin/uv pip check --python $r/.venv/bin/python
$r/.venv/bin/python -c "import jax,torch; from openpi.training import config; from openpi.policies import policy_config; print(\"CORE_IMPORT_OK\",config.get_config(\"pi05_libero\").name,jax.__version__,torch.__version__,flush=True)"
$r/tools/uv/bin/uv pip freeze --python $r/.venv/bin/python > $r/logs/installed_packages.txt
'
