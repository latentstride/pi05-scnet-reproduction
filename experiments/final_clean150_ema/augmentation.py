"""Training-only raw arm-state perturbation; absolute expert actions unchanged."""
import numpy as np
ARMS=np.array([0,1,2,3,4,5,7,8,9,10,11,12])

def perturb_samples(samples,index):
    rng=np.random.default_rng(np.random.SeedSequence([42,int(index),24011]))
    noise=rng.normal(0.,.003,(len(samples),12)).astype(np.float32)
    result=[]
    for sample,eps in zip(samples,noise,strict=True):
        item=dict(sample);item['state']=np.asarray(sample['state'],np.float32).copy()
        item['state'][ARMS]+=eps
        result.append(item)
    return result,noise

def verify_augmentation(ds,policy):
    windows=list(ds.indices('train',terminal=True))[:64]
    original=[ds.read('train',ep,row,terminal=True) for ep,row in windows]
    states=np.stack([s['state'].copy() for s in original])
    actions=np.stack([s['actions'].copy() for s in original])
    samples,noise=perturb_samples(original,1)
    max_error=0.;delta_error=0.
    for old,new in zip(original,samples,strict=True):
        np.testing.assert_array_equal(old['actions'],new['actions'])
        np.testing.assert_array_equal(old['state'][[6,13]],new['state'][[6,13]])
        before=policy.adapter.inputs(old)['actions'];after=policy.adapter.inputs(new)['actions']
        padded=np.pad(after,((0,0),(0,18)))
        restored=policy.adapter.outputs(new['state'],padded)
        err=float(np.max(np.abs(restored-new['actions'])))
        max_error=max(max_error,err)
        np.testing.assert_allclose(restored,new['actions'],rtol=2e-5,atol=2e-6)
        low=np.asarray(ds.norm_stats['actions']['q01'],np.float32)
        scale=np.asarray(ds.norm_stats['actions']['q99'],np.float32)-low+1e-6
        absolute=(after+1)*.5*scale+low
        absolute[:,ARMS]+=new['state'][ARMS]
        np.testing.assert_allclose(absolute,new['actions'],rtol=2e-5,atol=2e-6)
        delta_error=max(delta_error,float(np.max(np.abs(after[:,ARMS]-before[:,ARMS]))))
    np.testing.assert_array_equal(states,np.stack([s['state'] for s in original]))
    np.testing.assert_array_equal(actions,np.stack([s['actions'] for s in original]))
    assert delta_error>0
    return dict(samples=len(samples),noise_sigma_requested_rad=.003,observed_noise_std_rad=float(noise.std()),
        grippers_unchanged=True,absolute_targets_unchanged=True,source_samples_unmodified=True,
        native_adapter_and_auxiliary_target_roundtrip_max_error=max_error,
        normalized_delta_changed_max=delta_error,validation_and_inference_augmentation=False)
