"""Run only the new FK regression, without loading model/training dependencies."""
import json
from pathlib import Path
import sys
from g2_assets import load_arms
from k4_fk import verify_fk

root=Path(sys.argv[1]); bundle=root/'tools/a800-closed-loop'
safety=json.loads((bundle/'suite/docs/step6/arx_safety_profile.json').read_text())
arms=load_arms(bundle/'PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/assets',
              bundle/'pr5/research/g2/g2e-20261106-r1/release/asset-audit.json','ARX-X5',safety['calibration'])
print(json.dumps(verify_fk(arms)),flush=True)
