import unittest
import numpy as np
from g2_finger_force import centered_finger_force

class FingerForceTests(unittest.TestCase):
    def test_real_backend_disables_duplicate_drive_and_applies_named_bounded_effort(self):
        import tempfile
        from pathlib import Path
        from test_g2_robotwin_backend import fake_backend
        from test_g2_contact import world
        from g2_robotwin_backend import ARXBackend
        from g2_finger_force import MODE
        with tempfile.TemporaryDirectory() as tmp:
            b,adapter,states,_,q=fake_backend(Path(tmp));properties=[];forces=[]
            for side,(entity,_,grip) in enumerate(b._joints()):
                entity.get_qvel=lambda:np.zeros(8)
                entity.set_qf=lambda force,side=side:forces.append((side,force.copy()))
                states[side]['actual'][6:]=[.031,.029]
                states[side]['target'][6:]=0.
                for f,j in enumerate(grip):
                    j.set_drive_property=lambda **kw:properties.append(kw)
                    j.get_drive_target=lambda side=side,f=f:[states[side]['target'][6+f]]
            backend=ARXBackend(b.env,b.arms,world(adapter.arx),b.world_task,b.dt,b.initial,.01,1.,b.video,
                dict(stiffness=1000.,damping=200.,force_limit_n=2.,source='synthetic force allocation regression',mode=MODE))
            for side,(entity,_,grip) in enumerate(backend._joints()):backend._apply_forces(side,entity,grip)
            self.assertTrue(all(p['force_limit']==p['stiffness']==p['damping']==0 for p in properties))
            for side,force in forces:
                np.testing.assert_array_equal(force[:6],0.)
                self.assertLess(force[6],force[7]);self.assertTrue(np.all(abs(force[6:])<=2.))
            self.assertTrue(all(np.all(s['target'][6:]==0) for s in states))

    def test_saturated_closure_has_restoring_center_effort_with_same_force_cap(self):
        actual=np.array([.031,.029]);target=np.zeros(2)
        old=np.clip(1000*(target-actual),-2,2)
        np.testing.assert_array_equal(old,[-2.,-2.])
        new=centered_finger_force(actual,np.zeros(2),target,np.zeros(2),1000,200,2)
        self.assertTrue(np.all(abs(new)<=2.));self.assertLess(new[0],new[1])
        self.assertLess((actual[0]-actual[1])*(new[0]-new[1]),0.)
        np.testing.assert_array_equal(target,[0.,0.])
    def test_swapping_fingers_swaps_allocation_and_opening_still_works(self):
        rng=np.random.default_rng(42)
        for _ in range(100):
            q=rng.uniform(0,.044,2);v=rng.uniform(-.05,.05,2);target=rng.uniform(0,.044,2)
            a=centered_finger_force(q,v,target,np.zeros(2),1000,200,2)
            b=centered_finger_force(q[::-1],v[::-1],target[::-1],np.zeros(2),1000,200,2)
            np.testing.assert_array_equal(a[::-1],b)
            self.assertTrue(np.all(abs(a)<=2))
        np.testing.assert_array_equal(centered_finger_force([.03,.03],[0,0],[.044,.044],[0,0],1000,200,2),[2,2])
        self.assertTrue(np.all(centered_finger_force([.03,.03],[.01,.01],[.03,.03],[0,0],1000,200,2)<0))
