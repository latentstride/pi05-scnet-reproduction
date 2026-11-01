#!/usr/bin/env bash
# Run ONLY inside the independent research sandbox, with /public bound.
set -euo pipefail
: "${PI05_ROOT:?}"
r=/opt/robotwin
mkdir -p "$r"/{cache,src,apt/lists/partial,apt/archives/partial,sysroot,logs,tmp,python}
export UV_CACHE_DIR="$r/cache/uv" UV_PYTHON_INSTALL_DIR="$r/python"
export UV_PYTHON_BIN_DIR="$r/bin" UV_PYTHON_INSTALL_BIN=0
export PIP_CACHE_DIR="$r/cache/pip" TMPDIR="$r/tmp"
export XDG_CACHE_HOME="$r/cache/xdg" TORCH_EXTENSIONS_DIR="$r/cache/torch_extensions"
export CUDA_CACHE_PATH="$r/cache/cuda" CUDA_HOME=/usr/local/cuda
export MAX_JOBS=1 TORCH_CUDA_ARCH_LIST='8.9+PTX'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$CUDA_HOME/bin:$PATH"
u="$PI05_ROOT/tools/uv/bin/uv"
stage() { printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$1"; }
trap 'c=$?; if (( c )); then echo "BUILD_FAILED exit=$c"; fi' EXIT
stage system-packages
aptopts=(-o "Dir::State::lists=$r/apt/lists" -o "Dir::Cache::archives=$r/apt/archives" -o 'Debug::NoLocking=true')
apt-get "${aptopts[@]}" update
cd "$r/apt/archives"
apt-get "${aptopts[@]}" download curl libcurl4 libnghttp2-14 libpsl5 libssh-4 \
 librtmp1 libldap-2.5-0 libbrotli1 libgssapi-krb5-2 libkrb5-3 libk5crypto3 \
 libkeyutils1 libkrb5support0 libsasl2-2 libunistring2 libidn2-0 libgnutls30 \
 libnettle8 libhogweed6 libp11-kit0 libtasn1-6 libgmp10 libcom-err2 \
 libvulkan1 vulkan-tools libegl1 libgl1 libglvnd0 \
 libglx0 libopengl0 libxext6 libx11-6 libx11-xcb1 libxcb1 libxau6 libxdmcp6 \
 libxrender1 libsm6 libice6 libbsd0 libmd0 libc++1-14 libc++abi1-14 libunwind-14 \
 libglib2.0-0 libgomp1 libxrandr2 libxfixes3 \
 libxcursor1 libxi6 libxinerama1 libxkbcommon0 libwayland-client0 \
 libwayland-cursor0 libwayland-egl1 libdrm2 libexpat1 libffi8 libfontconfig1
for deb in ./*.deb; do dpkg-deb -x "$deb" "$r/sysroot"; done
export LD_LIBRARY_PATH="$r/sysroot/usr/lib/x86_64-linux-gnu:$r/sysroot/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$r/sysroot/usr/bin:$PATH"
stage python
"$u" --no-config python install 3.10.19
if ! test -x "$r/venv/bin/python"; then
 "$u" --no-config venv --python "$r/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10" "$r/venv"
fi
p="$r/venv/bin/python"
stage python-core-dependencies
"$u" --no-config pip install --python "$p" --index-url https://pypi.org/simple \
 setuptools==69.5.1 wheel setuptools-scm ninja packaging pybind11 \
 numpy==1.26.4 scipy==1.10.1 pyyaml==6.0.2 pillow==10.4.0
"$u" --no-config pip install --python "$p" --index-url https://download.pytorch.org/whl/cu121 \
 torch==2.4.1 torchvision==0.19.1
stage robotwin-python-dependencies
"$u" --no-config pip install --python "$p" --index-url https://pypi.org/simple \
 -c "$PI05_ROOT/research/robotwin/env-build/constraints.txt" \
 sapien==3.0.0b1 transforms3d==0.4.2 mplib==0.2.1 gymnasium==0.29.1 \
 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr==2.18.3 \
 numcodecs==0.13.1 openai huggingface_hub==0.25.0 h5py azure==4.0.0 \
 azure-ai-inference 'pyglet<2' wandb moviepy==1.0.3 termcolor av matplotlib \
 ffmpeg imageio-ffmpeg networkx numpy-quaternion yourdfpy warp-lang==1.12.0 \
 tqdm importlib_resources scikit-image iopath fvcore
fetch_source() {
 local repo=$1 revision=$2 destination=$3
 if ! test -d "$destination"; then
  curl --fail --location --retry 2 --connect-timeout 20 --max-time 240 \
   "https://codeload.github.com/$repo/tar.gz/$revision" -o "$r/src/$(basename "$destination").tar.gz"
  mkdir -p "$destination"
  tar -xzf "$r/src/$(basename "$destination").tar.gz" --strip-components=1 -C "$destination"
 fi
}
stage source-downloads
fetch_source NVlabs/curobo d64c4b005459db10c5dd867d8b30a87d5bda9bdb "$r/src/curobo"
fetch_source facebookresearch/pytorch3d 75ebeeaea0908c5527e7b1e305fbc7681382db47 "$r/src/pytorch3d"
fetch_source RoboTwin-Platform/RoboTwin 13c3c47ff4312dd62484bcd51be034af55c062d1 "$r/src/RoboTwin"
stage curobo-cuda-build
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.8
"$u" --no-config pip install --python "$p" --no-build-isolation --no-deps "$r/src/curobo"
unset SETUPTOOLS_SCM_PRETEND_VERSION
stage pytorch3d-build
FORCE_CUDA=1 "$u" --no-config pip install --python "$p" --no-build-isolation --no-deps "$r/src/pytorch3d"
stage validation
cat > /.singularity.d/env/99-robotwin.sh <<'ENV'
export PATH=/opt/robotwin/venv/bin:/opt/robotwin/sysroot/usr/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/robotwin/sysroot/usr/lib/x86_64-linux-gnu:/opt/robotwin/sysroot/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/opt/robotwin/sysroot/etc/vulkan/icd.d/nvidia_icd.json
export XDG_CACHE_HOME=/opt/robotwin/cache/xdg
export TORCH_EXTENSIONS_DIR=/opt/robotwin/cache/torch_extensions
export CUDA_CACHE_PATH=/opt/robotwin/cache/cuda
export PYTHONPYCACHEPREFIX=/opt/robotwin/cache/pycache
export TMPDIR=/opt/robotwin/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ENV
mkdir -p "$r/sysroot/etc/vulkan/icd.d"
# Vendor library is supplied later by Apptainer --nv. This is discovery
# configuration, not a bundled driver and not graphics acceptance.
cat > "$r/sysroot/etc/vulkan/icd.d/nvidia_icd.json" <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.0.0"}}
JSON
mkdir -p /etc/vulkan/icd.d
cp "$r/sysroot/etc/vulkan/icd.d/nvidia_icd.json" /etc/vulkan/icd.d/nvidia_icd.json
"$p" - <<'PY'
from pathlib import Path
import sapien
p = Path(sapien.__file__).parent/'wrapper/urdf_loader.py'
text = p.read_text()
text = text.replace('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")')
p.write_text(text)
PY
"$u" --no-config pip check --python "$p"
"$u" --no-config pip freeze --python "$p" > "$r/logs/requirements.installed.txt"
"$p" "$PI05_ROOT/scripts/validate_robotwin_env.py" --output "$r/logs/cpu_validation.json"
stage finished
touch "$r/logs/BUILD_COMPLETE"
