#!/usr/bin/env bash
#SBATCH --job-name=cpf-rw-graphics
#SBATCH --partition=ksagnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/robotwin-graphics-%j.out
set -euo pipefail
ulimit -c 0
root="$HOME/code_cpf/pi05"
image="$root/research/robotwin/robotwin-cu122"
a=${APPTAINER_BIN:-/public/software/apps/apptainer/1.3.4/bin/apptainer}
test -f "$image/opt/robotwin/logs/BUILD_COMPLETE"

# Run the image's userspace tool as a native host process. The explicit
# interpreter and library paths keep it independent of the host's glibc
# version and of vulkaninfo availability on the compute node.
host_vulkaninfo() {
 local sysroot="$image/opt/robotwin/sysroot"
 local arch=x86_64-linux-gnu
 local libs="$sysroot/usr/lib/$arch:$sysroot/lib/$arch:$image/usr/lib/$arch:$image/lib/$arch"
 if [[ -n ${host_icd_library:-} ]]; then
  libs="$libs:$(dirname "$host_icd_library")"
 fi
 timeout --kill-after=5 45 env -u DISPLAY -u WAYLAND_DISPLAY -u LD_PRELOAD -u LD_LIBRARY_PATH \
  "$image/lib/$arch/ld-linux-x86-64.so.2" --library-path "$libs" \
  "$sysroot/usr/bin/vulkaninfo" "$@"
}
if [[ ${1:-} == --host-tool-check ]]; then
 # This vulkaninfo release prints usage and exits with status 1 for --help.
 help_status=0
 help_output=$(host_vulkaninfo --help 2>&1) || help_status=$?
 printf '%s\n' "$help_output"
 [[ "$help_status" -le 1 && "$help_output" == *"USAGE:"* && "$help_output" == *"--summary"* ]]
 exit
fi

cd "${SLURM_SUBMIT_DIR:?Submit from personal project root}"
test "$PWD" = "$root"
: "${SLURM_JOB_ID:?Must be an allocated Slurm GPU job}"
out="$root/research/robotwin/graphics-${SLURM_JOB_ID}"
mkdir -p "$out"
failed_stage=allocation
trap 'code=$?; if ((code)); then printf "%s exit=%s\n" "$failed_stage" "$code" > "$out/FAILED"; fi' EXIT
stage() { failed_stage=$1; printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1"; }
stage nvidia_visibility
{
 printf 'SLURM_STEP_GPUS=%s\n' "${SLURM_STEP_GPUS:-}"
 printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-}"
 printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
} > "$out/gpu_visibility.txt"
# CUDA honors the job's visibility mask. Use the visible device's PCI address
# because Slurm, CUDA and NVML numeric indices can refer to different devices.
if ! gpu_pci=$(timeout 15 python3 - 2>> "$out/gpu_visibility.txt" <<'PY'
import ctypes as c

cuda = c.CDLL("libcuda.so.1")


def check(operation, *args):
    code = getattr(cuda, operation)(*args)
    if code:
        raise RuntimeError(f"{operation} failed with CUDA status {code}")


check("cuInit", 0)
count = c.c_int()
check("cuDeviceGetCount", c.byref(count))
if count.value != 1:
    raise RuntimeError(f"Expected one visible CUDA GPU, found {count.value}")
device = c.c_int()
check("cuDeviceGet", c.byref(device), 0)
pci = c.create_string_buffer(32)
check("cuDeviceGetPCIBusId", pci, len(pci), device)
print(pci.value.decode("ascii"))
PY
); then
 timeout 10 nvidia-smi -L >> "$out/gpu_visibility.txt" 2>&1 || true
 cat "$out/gpu_visibility.txt" >&2
 exit 1
fi
printf 'CUDA_PCI=%s\n' "$gpu_pci" >> "$out/gpu_visibility.txt"
if ! timeout 20 nvidia-smi -i "$gpu_pci" --query-gpu=uuid,index,name,pci.bus_id,driver_version \
 --format=csv,noheader > "$out/allocated_nvidia.csv" 2>> "$out/gpu_visibility.txt"; then
 cat "$out/allocated_nvidia.csv" >> "$out/gpu_visibility.txt"
 timeout 10 nvidia-smi -L >> "$out/gpu_visibility.txt" 2>&1 || true
 cat "$out/gpu_visibility.txt" >&2
 exit 1
fi
read -r gpu_uuid < <(cut -d, -f1 "$out/allocated_nvidia.csv")
test "$(wc -l < "$out/allocated_nvidia.csv")" -eq 1
[[ "$gpu_uuid" =~ ^GPU-[0-9a-fA-F-]+$ ]]
for assigned in "${SLURM_STEP_GPUS:-}" "${SLURM_JOB_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"; do
 if [[ "$assigned" == GPU-* && "${assigned,,}" != "${gpu_uuid,,}" ]]; then
  echo "CUDA device UUID disagrees with the scheduler-provided UUID" >&2
  exit 1
 fi
done
stage graphics_driver_discovery
# Only bind installed host NVIDIA libraries, read-only; never install a driver.
python3 - "$out" "$image" <<'PY'
from pathlib import Path
import csv,json,os,re,shutil,subprocess,sys
out,image=map(Path,sys.argv[1:])
row=next(csv.reader((out/'allocated_nvidia.csv').read_text().splitlines()))
uuid,index,name,bus,driver=[x.strip() for x in row]
domain,bus_id,rest=bus.lower().split(':')
pci='%04x:%s:%s'%(int(domain,16),bus_id,rest)
(out/'allocation.json').write_text(json.dumps({
 'gpu_uuid':uuid,'index':index,'name':name,'pci_bus_id':pci,'driver_version':driver,
 'slurm_job_gpus':os.environ.get('SLURM_JOB_GPUS'),
 'cuda_visible_original':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2))
icds=[]
candidates=[]
for directory in ['/etc/vulkan/icd.d','/usr/share/vulkan/icd.d','/usr/local/share/vulkan/icd.d']:
 candidates.extend(sorted(Path(directory).glob('*.json')))
for variable in ['VK_ICD_FILENAMES','VK_DRIVER_FILES']:
 candidates.extend(Path(x) for x in os.environ.get(variable,'').split(':') if x)
for p in candidates:
 try:
  doc=json.loads(p.read_text()); lib=doc.get('ICD',{}).get('library_path','')
  if 'nvidia' in lib.lower() and doc['ICD'].get('library_arch') != '32': icds.append((p,doc))
 except (ValueError,OSError): pass
assert icds, 'GPU host has no NVIDIA Vulkan ICD; administrator graphics setup required'
icd,doc=icds[0]
ldconfig=shutil.which('ldconfig') or '/sbin/ldconfig'
listing=subprocess.check_output([ldconfig,'-p'],universal_newlines=True)
libraries={}
for line in listing.splitlines():
 if 'x86-64' not in line or '=>' not in line: continue
 soname=line.split()[0]; source=line.rsplit('=>',1)[1].strip()
 if re.match(r'lib(?:GLX_nvidia|EGL_nvidia|nvidia-[A-Za-z0-9_-]+)\.so',soname) and Path(source).is_file():
  libraries.setdefault(soname,source)
icd_library=doc['ICD']['library_path']; key=Path(icd_library).name
if Path(icd_library).is_absolute() and Path(icd_library).is_file(): libraries[key]=icd_library
assert key in libraries, 'NVIDIA ICD library missing from host linker cache'
inside='/opt/robotwin/host-driver-'+out.name
destination=image/inside.lstrip('/')
destination.mkdir(parents=True,exist_ok=True)
bindings=[]
for soname,source in sorted(libraries.items()):
 (destination/soname).touch(exist_ok=True)
 bindings.append(source+'\t'+inside+'/'+soname)
(out/'driver_binds.tsv').write_text('\n'.join(bindings)+'\n')
(out/'host_icd_path.txt').write_text(str(icd)+'\n')
(out/'host_icd_library.txt').write_text(libraries[key]+'\n')
(out/'driver_inside_path.txt').write_text(inside+'\n')
doc['ICD']['library_path']=libraries[key]
(out/'host_nvidia_icd.json').write_text(json.dumps(doc,indent=2))
doc['ICD']['library_path']=inside+'/'+key
(out/'container_nvidia_icd.json').write_text(json.dumps(doc,indent=2))
print('allocated_gpu',name,'driver',driver,'graphics_libraries',len(bindings),flush=True)
PY
stage host_vulkan
host_icd_library=$(cat "$out/host_icd_library.txt")
VK_DRIVER_FILES="$out/host_nvidia_icd.json" VK_ICD_FILENAMES="$out/host_nvidia_icd.json" \
 host_vulkaninfo > "$out/host_vulkan.txt" 2>&1
driver_inside=$(cat "$out/driver_inside_path.txt")
gpu_pci=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pci_bus_id"])' "$out/allocation.json")
options=(--writable --nv --cleanenv --bind /public:/public --pwd "$root")
while IFS=$'\t' read -r source destination; do
 options+=(--bind "$source:$destination:ro")
done < "$out/driver_binds.tsv"
inside() {
 local seconds=$1
 shift
 timeout --kill-after=5 "$seconds" "$a" exec "${options[@]}" "$image" env \
  PATH=/opt/robotwin/venv/bin:/opt/robotwin/sysroot/usr/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH="$driver_inside:/opt/robotwin/sysroot/usr/lib/x86_64-linux-gnu:/opt/robotwin/sysroot/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/.singularity.d/libs" \
  PI05_ROOT="$root" CUDA_VISIBLE_DEVICES="$gpu_uuid" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  ROBOTWIN_EXPECTED_PCI="$gpu_pci" DISPLAY= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VK_DRIVER_FILES="$out/container_nvidia_icd.json" VK_ICD_FILENAMES="$out/container_nvidia_icd.json" \
  PYTHONUNBUFFERED=1 \
  XDG_CACHE_HOME=/opt/robotwin/cache/xdg TORCH_EXTENSIONS_DIR=/opt/robotwin/cache/torch_extensions \
  CUDA_CACHE_PATH=/opt/robotwin/cache/cuda TMPDIR=/opt/robotwin/tmp \
  PYTHONPYCACHEPREFIX=/opt/robotwin/cache/pycache "$@"
}
stage container_vulkan
inside 45 vulkaninfo > "$out/container_vulkan.txt" 2>&1
stage vulkan_identity
python3 - "$out" <<'PY'
from pathlib import Path
import sys,re,json
p=Path(sys.argv[1]); allocation=json.loads((p/'allocation.json').read_text())
expected=allocation['gpu_uuid'][4:].lower().replace('-','')
seen=[]
for name in ['host','container']:
 text=(p/(name+'_vulkan.txt')).read_text(); uuids=set()
 for line in text.splitlines():
  if re.search(r'\bdeviceUUID\s*=',line):
   value=re.sub('0x','',line.split('=',1)[1],flags=re.I)
   value=re.sub('[^0-9a-f]','',value.lower())
   if len(value)==32: uuids.add(value)
 assert expected in uuids, name+': Vulkan does not enumerate the allocated NVIDIA UUID'
 seen.append(expected)
(p/'vulkan_identity.json').write_text(json.dumps({'pass':True,'allocated_uuid':allocation['gpu_uuid'],
 'host_and_container_same_allocated_gpu':seen[0]==seen[1]},indent=2))
PY
stage sapien_render
inside 150 python scripts/robotwin_gpu_smoke.py --stage render --output "$out" > "$out/sapien.log" 2>&1
stage curobo_plan
inside 240 python scripts/robotwin_gpu_smoke.py --stage plan --output "$out" > "$out/curobo.log" 2>&1
stage summary
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); result={k:json.loads((p/(k+'.json')).read_text()) for k in ['vulkan_identity','render','plan']}
assert all(x['pass'] for x in result.values())
(p/'acceptance.json').write_text(json.dumps({'passed':True,'job_id':p.name.split('-')[-1],
 'checks':result,'policy_weights_loaded':False},indent=2))
print('ROBOTWIN_GRAPHICS_GATE_PASS',flush=True)
PY
