"""Serve the official policy on loopback for the evaluation client."""
import argparse
import json
import pathlib
import time

import jax
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
args = p.parse_args()
assert any(d.platform == 'gpu' for d in jax.devices()), jax.devices()
t = time.perf_counter()
policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), pathlib.Path('cache/checkpoints/pi05_libero'))
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'server_load.json').write_text(json.dumps({'load_seconds': time.perf_counter() - t, 'policy': 'pi05_libero', 'port': args.port}))
print('POLICY_READY', flush=True)
websocket_policy_server.WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port).serve_forever()
