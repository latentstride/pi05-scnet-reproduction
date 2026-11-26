"""Original expert labels; half uniform, half cycling six event windows."""
import numpy as np
from scripts.step5_arx.sampling import grasp_events

PHASES=('approach','left_close','lift','right_approach','handover_dual_grip_and_release','right_place')

def events(targets):
    close,lift=grasp_events(targets)
    def first(condition,start,name):
        idx=np.flatnonzero(condition & (np.arange(len(condition))>=start))
        if not len(idx):raise ValueError('Missing expert event '+name)
        return int(idx[0])
    left_closed=first(targets[:,6]<=1e-4,close,'left_closed')
    right_delta=np.r_[0,np.diff(targets[:,13])]
    right_close=first(right_delta < -1e-4,left_closed,'right_close')
    right_closed=first(targets[:,13]<=1e-4,right_close,'right_closed')
    left_delta=np.r_[0,np.diff(targets[:,6])]
    left_release=first(left_delta>1e-4,right_closed,'left_release_after_right_closed')
    right_release=first(right_delta>1e-4,left_release,'right_release_at_place')
    return dict(left_close=close,lift=lift,left_closed=left_closed,right_close=right_close,
                right_closed=right_closed,left_release=left_release,right_release=right_release)

class SixPhaseSampler:
    def __init__(self,ds,split='train'):
        self.phases={name:[] for name in PHASES}
        self.events={}
        self.episode_pools={name:{} for name in PHASES}
        for ep,g in sorted(ds.groups[split].items()):
            targets=g['drive_target'][:];e=events(targets);self.events[ep]=e
            close,lift,rc,lr,rr=(e[k] for k in ('left_close','lift','right_close','left_release','right_release'))
            ranges=[range(close-8,close-4),range(close-4,close),range(lift-4,lift),
                    range(rc-8,rc-4),list(range(rc-4,rc))+list(range(lr-4,lr)),range(lr,rr+4)]
            for phase,indices in zip(PHASES,ranges):
                pool=sorted(set((ep,int(t)) for t in indices if 0<=t<len(targets)-1))
                if not pool:raise ValueError((ep,phase,e))
                self.episode_pools[phase][ep]=pool;self.phases[phase].extend(pool)
    def batch(self,seed,step,size,terminal_windows):
        rng=np.random.default_rng(np.random.SeedSequence([seed,step]))
        if step%5:
            return [terminal_windows[i] for i in rng.choice(len(terminal_windows),size=size,replace=False)]
        phase=PHASES[(step//5-1)%6]
        # Episode-uniform selection prevents long placement spans dominating.
        eps=sorted(self.episode_pools[phase])
        selected=rng.choice(eps,size=size,replace=False)
        return [self.episode_pools[phase][int(ep)][int(rng.integers(len(self.episode_pools[phase][int(ep)])))] for ep in selected]
    def manifest(self):
        return dict(events=self.events,phases={k:dict(windows=len(v),episodes=len(self.episode_pools[k])) for k,v in self.phases.items()},
                    rule='80% uniform terminal windows; every fifth microbatch round-robin six phases, episode-uniform; original targets unchanged',
                    windows='approach/left_close/lift unchanged; right_approach right_close-8:-4; handover union right_close-4:0 and left_release-4:0; right_place left_release:right_release+4',
                    time_semantics='saved-row indices; no assumption of fixed physical time')
