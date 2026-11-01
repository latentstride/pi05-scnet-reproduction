#!/usr/bin/env bash
# Download official Ubuntu packages and unpack inside this project; no host installation.
set -euo pipefail
cd "$(dirname "$0")/.."
r=$PWD
mkdir -p sysroot/lists/partial sysroot/archives/partial sysroot/root
/public/software/apps/apptainer/1.3.4/bin/apptainer exec --writable --cleanenv --bind /public:/public \
  containers/pi05-devel env PI05_ROOT="$r" bash -c '
set -e
r=$PI05_ROOT
cp /var/lib/dpkg/status "$r/sysroot/status"
opts=(-o Dir::State::lists="$r/sysroot/lists" -o Dir::Cache::archives="$r/sysroot/archives" -o Dir::State::status="$r/sysroot/status" -o Debug::NoLocking=1)
apt-get "${opts[@]}" update
apt-get "${opts[@]}" --download-only --no-install-recommends -y install libegl1 libgl1 libglib2.0-0 libosmesa6
for f in "$r"/sysroot/archives/*.deb; do dpkg-deb -x "$f" "$r/sysroot/root"; done
echo RENDER_LIBRARIES_EXTRACTED
'
