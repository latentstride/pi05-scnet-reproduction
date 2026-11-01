#!/usr/bin/env bash
set -euo pipefail
r=${PI05_ROOT:?}
u=$r/tools/uv/bin/uv
s=$r/source/openpi-$(cat "$r/source/OPENPI_COMMIT")
test -x "$r/.venv-libero/bin/python" || "$u" venv --python 3.8 "$r/.venv-libero"
"$u" pip install --python "$r/.venv-libero/bin/python" \
  -r "$s/examples/libero/requirements.txt" -r "$s/third_party/libero/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
"$u" pip install --python "$r/.venv-libero/bin/python" --no-deps \
  -e "$s/packages/openpi-client" -e "$s/third_party/libero"
"$u" pip check --python "$r/.venv-libero/bin/python"
"$u" pip freeze --python "$r/.venv-libero/bin/python" > "$r/logs/libero_packages.txt"
