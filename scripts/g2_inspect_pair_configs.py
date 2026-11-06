import json,sys
from hashlib import sha256
from pathlib import Path
from g2_task_runtime import validate_config, inspect_assets, verify_runtime
configs, pace, audit, source, output = map(Path, sys.argv[1:])
verify_runtime(pace,json.loads(source.read_text()))
manifest=json.loads((configs/'diagnostic-roster.json').read_text())
rows=[]
for row in manifest['episodes']:
    path=configs/row['config']
    assert sha256(path.read_bytes()).hexdigest()==row['config_sha256']
    config=json.loads(path.read_text())
    validate_config(config)
    inspection=inspect_assets(config,pace/'assets',audit)
    rows.append(dict(episode_id=row['episode_id'],config_sha256=row['config_sha256'],inspection=inspection))
output.write_text(json.dumps(dict(source_audit_verified=True, configs=rows,renderer_created=False,gpu_used=False),indent=2)+'\n')
print('CONFIG_INSPECT_OK',len(rows),flush=True)
