import json,sys
from pathlib import Path
import ema_model as model
from dataset150 import Dataset
from scripts.step5_arx.serve import Session
from openpi.serving import websocket_policy_server
root,experiment,output=map(Path,sys.argv[1:4]);step=int(sys.argv[4])
graph,p,f,_,_,report=model.load(root,with_optimizer=False)
saved=model.restore(experiment/f'checkpoints/step-{step:04d}',dict(ema=p.to_pure_dict()));p.replace_by_pure_dict(saved['ema']);del saved
with Dataset(root/'data/step5-arx-clean150-newnorm-20261126-v1.h5') as ds:policy=model.NativePolicy(ds.norm_stats);norm=ds.norm_stats
metadata=dict(checkpoint=str(experiment/f'checkpoints/step-{step:04d}'),weights='EMA',ema_decay=.99,denoise_steps=10,norm_stats=norm)
(output/'model.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('EMA_MODEL_READY',step,flush=True)
websocket_policy_server.WebsocketPolicyServer(Session(policy,model.sampler(graph,adapted=True),(p,f),42),host='127.0.0.1',port=52060,metadata=metadata).serve_forever()
