"""Derive a G2 graphics probe from the pinned G1 source."""
from pathlib import Path
import sys


def make(source, target):
    text = source.read_text()
    old = '[[ -z "$mode" || "$mode" == --g1 ]]'
    if text.count(old) != 1:
        raise ValueError('G1 graphics entry layout changed')
    text = text.replace(old,'[[ "$mode" == --g2 ]]\nrelease=${2:?Expected fixed release directory}\ng2_out=${3:?Expected graphics output path}')
    text = text.replace('mkdir -p "$out"','out="$g2_out"\nmkdir -p "$out"',1)
    text = text.replace('local probe="$root/scripts/robotwin_graphics_driver.py"', 'local probe="$release/scripts/robotwin_graphics_driver.py"')
    marker = ' if [[ "$mode" == --g1 ]]; then\n  stage g1_graphics_gate'
    prefix, _ = text.split(marker,1)
    if 'container_vulkan' not in prefix or 'run_check vulkan_identity' not in prefix:
        raise ValueError('G1 graphics verification layout changed')
    script = prefix + ''' test -f "$out/vulkan_identity.json"
 make_sim_command
 run_check sapien_render inside 110 python "$release/scripts/robotwin_gpu_smoke.py" --stage render --output "$out" > "$out/sapien.log" 2>&1
 python3 - "$out/sim-launch.json" "$gpu_uuid" "${sim_cmd[@]}" <<'JSON'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(gpu_uuid=sys.argv[2],sim_argv=sys.argv[3:]),indent=2)+'\\n')
JSON
 exit
fi
exit 1
'''
    target.write_text(script)


if __name__=='__main__':
    make(*map(Path,sys.argv[1:]))
