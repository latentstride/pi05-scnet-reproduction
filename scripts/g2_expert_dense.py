"""Recover every physics-step control update from the official cached paths."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from g2_audit_expert_metadata import NumpyOnlyUnpickler,sha

STAGES=[('L',0),('L',1),('LG',0),('L',2),('L',3),('L',4),('R',0),('R',1),('RG',0),('LG',1),('L',5),('B',(6,2)),('R',3),('RG',1)]
def dense(paths,initial):
    q=initial.copy();v=np.zeros((2,6));rows=[q.copy()];velocities=[v.copy()];active=[np.array([False,False])];eps=[np.zeros(2)];stage=[-1];segments=[]
    for si,(kind,index) in enumerate(STAGES):
        begin=len(rows)-1
        if kind=='B':
            lp=paths['left_joint_path'][index[0]];rp=paths['right_joint_path'][index[1]];li=ri=0
            while li<len(lp['position']) or ri<len(rp['position']):
                if li<len(lp['position']) and li/len(lp['position'])<=ri/len(rp['position']):
                    q[:6]=lp['position'][li];v[0]=lp['velocity'][li];li+=1
                if ri<len(rp['position']) and ri/len(rp['position'])<=li/len(lp['position']):
                    q[7:13]=rp['position'][ri];v[1]=rp['velocity'][ri];ri+=1
                rows.append(q.copy());velocities.append(v.copy());active.append(np.array([False,False]));eps.append(np.zeros(2));stage.append(si)
        else:
            fg=kind.endswith('G');side=0 if kind.startswith('L') else 1
            if fg:
                j=6 if side==0 else 13;delta=(index-q[j])/200
                values=np.r_[np.linspace(q[j],index,200),np.repeat(index,100)]
            else:plan=paths['left_joint_path' if side==0 else 'right_joint_path'][index];values=plan['position']
            for i,value in enumerate(values):
                ac=np.array([False,False]);e=np.zeros(2)
                if fg:q[j]=value;ac[side]=True;e[side]=delta
                else:q[slice(0,6) if side==0 else slice(7,13)]=value;v[side]=plan['velocity'][i]
                rows.append(q.copy());velocities.append(v.copy());active.append(ac);eps.append(e);stage.append(si)
        segments.append(dict(kind=kind,path_index=index,first_tick=begin,last_tick=len(rows)-1))
    return dict(target=np.array(rows),arm_velocity=np.array(velocities),gripper_active=np.array(active),gripper_eps=np.array(eps),stage=np.array(stage)),segments

def packed_controls(data):
    """Losslessly encode float32 planner arrays and float64 gripper requests."""
    out={}
    for key in ('target','arm_velocity'):
        a=data[key].astype(np.float32);bits=a.reshape(len(a),-1).view(np.uint32)
        delta=bits.copy();delta[1:]=np.bitwise_xor(bits[1:],bits[:-1])
        out[key+'_delta']=delta.view(np.uint8).T.copy()
    for key in ('gripper_active','gripper_eps','stage','frame_ticks'):out[key]=data[key]
    out['gripper_norm']=data['target'][:,[6,13]]
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];cache=root/'research/g2/expert-cache-uncached-20261106';ep=a.episode
    file=cache/f'episode{ep}.pkl';manifest=json.loads((cache/'provenance.json').read_text());assert sha(file)==next(m['sha256'] for m in manifest['members'] if m['file']==file.name)
    with file.open('rb') as f:paths=NumpyOnlyUnpickler(f).load()
    with h5py.File(root/f'research/robotwin/data_probe/aloha-agilex/episode{ep}.hdf5') as f:
        actual=f['joint_action/vector'][:];expected_endpose=np.stack([f['endpose/left_endpose'][:],f['endpose/right_endpose'][:]],axis=1)
    sparse=np.load(root/f'research/g2/expert-metadata-20261106/episode{ep}_derived_timing.npz');data,segments=dense(paths,actual[0]);ticks=sparse['physics_tick']
    np.testing.assert_array_equal(data['target'][ticks],actual)
    a.output.mkdir(exist_ok=False,parents=True);np.savez_compressed(a.output/'trace.npz',**data,frame_ticks=ticks,expected_endpose=expected_endpose)
    np.savez_compressed(a.output/'runtime-trace.npz',**packed_controls(dict(data,frame_ticks=ticks)))
    entry=json.loads((root/'research/g2/expert-metadata-20261106/audit.json').read_text())['episodes'][ep]
    (a.output/'input.json').write_text(json.dumps(dict(episode=ep,seed=entry['seed'],spawn_pose=entry['spawn_pose'],segments=segments,physics_dt_s=.004,frames=len(ticks),physics_steps=len(data['target'])-1,planner_sha256=sha(file),all_recorded_targets_exact=True),indent=2)+'\n')
    print('dense physics steps',len(data['target'])-1,'frames',len(ticks),'bytes',(a.output/'trace.npz').stat().st_size)
