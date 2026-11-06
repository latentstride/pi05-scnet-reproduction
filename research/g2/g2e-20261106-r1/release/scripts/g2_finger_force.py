"""Bounded aperture/center force allocation for two independently measured fingers."""
import numpy as np
from g2_adapter import positive
from robotwin_control_geometry import finite_array

MODE='bounded_aperture_center_force_v1'

def centered_finger_force(position,velocity,target,target_velocity,stiffness,damping,cap):
    position=finite_array(position,(2,));velocity=finite_array(velocity,(2,))
    target=finite_array(target,(2,));target_velocity=finite_array(target_velocity,(2,))
    stiffness=positive(stiffness);damping=positive(damping);cap=positive(cap)
    proportional=stiffness*(target-position)+damping*(target_velocity-velocity)
    # Clip common closure effort BEFORE adding differential center feedback.
    # Independent saturated PD drives otherwise have zero center stiffness.
    common=float(np.clip(np.mean(proportional),-cap,cap))
    center=float((proportional[0]-proportional[1])/2)
    force=np.clip(common+np.array([center,-center]),-cap,cap)
    return force
