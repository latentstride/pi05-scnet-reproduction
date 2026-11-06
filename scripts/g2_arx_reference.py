"""Retiming only for the sampled, verified FK/IK expert reference; no collision verdict."""
import numpy as np

def retime_reference(q,grip,source_ticks,dt=.004,arm_acceleration=1.):
    q=np.asarray(q,float);grip=np.asarray(grip,float);source_ticks=np.asarray(source_ticks,int)
    if q.shape!=(len(grip),2,6) or grip.shape!=(len(q),2) or len(source_ticks)!=len(q):raise ValueError('reference shapes')
    if not np.isfinite(q).all() or not np.isfinite(grip).all() or np.any((grip<0)|(grip>1)) or np.any(np.diff(source_ticks)<0):raise ValueError('invalid reference')
    if not np.isfinite(arm_acceleration) or not 0<arm_acceleration<=1. or not np.isfinite(dt) or dt<=0:
        raise ValueError('invalid reference timing limits')
    positions=np.concatenate([q,np.repeat((.044*grip)[:,:,None],2,axis=2)],axis=2)
    speed=np.array([[.5]*6+[.03]*2]*2);accel=np.array([[arm_acceleration]*6+[.1]*2]*2)
    path=[positions[0]];velocity=[np.zeros((2,8))];events=[0];max_v=np.zeros((2,8));max_a=np.zeros((2,8))
    for i,(start,end) in enumerate(zip(positions[:-1],positions[1:]),1):
        d=end-start;duration=max(float(np.max(1.875*abs(d)/speed)),float(np.sqrt(np.max(10/np.sqrt(3)*abs(d)/accel))),(source_ticks[i]-source_ticks[i-1])*.004)
        n=int(np.ceil(duration/dt))
        if n:
            u=np.arange(1,n+1)/n;s=u**3*(10+u*(-15+6*u));v=30*u**2*(1-u)**2/(n*dt)
            path.extend(start+s[:,None,None]*d);velocity.extend(v[:,None,None]*d)
            max_v=np.maximum(max_v,1.875*abs(d)/(n*dt));max_a=np.maximum(max_a,10/np.sqrt(3)*abs(d)/(n*dt)**2)
        events.append(len(path)-1)
    path=np.array(path);velocity=np.array(velocity);target=np.concatenate([np.c_[path[:,0,:6],path[:,0,6]/.044],np.c_[path[:,1,:6],path[:,1,6]/.044]],axis=1)
    # The drive scalar is an intent; measured fingers may stop on the object.
    return dict(target=target,arm_velocity=velocity[:,:,:6],gripper_active=np.ones((len(path),2),bool),gripper_eps=np.zeros((len(path),2)),finger_velocity=velocity[:,:,6],frame_ticks=np.array(events),stage=np.zeros(len(path),int)),dict(max_command_velocity=max_v.tolist(),max_command_acceleration=max_a.tolist(),arm_acceleration_limit_rad_s2=arm_acceleration,original_source_duration_s=float(source_ticks[-1]*.004),retimed_duration_s=(len(path)-1)*dt,frames=len(q),ticks=len(path)-1,collision_checked=False,method='quintic_stop_at_saved_expert_waypoints',limits_source='G2 experiment-imposed simulation command bounds v1; not manufacturer ratings')
