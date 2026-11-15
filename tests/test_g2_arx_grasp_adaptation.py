import numpy as np

from g2_arx_grasp_adaptation import FirstLeftGrasp, JawGeometry
from robotwin_control_geometry import inverse_pose


def fixture():
    directions = np.array([[0., -1., 0.], [0., 1., 0.]])
    surfaces = np.array([[[.02, -.001, -.008], [.038, -.001, .01]],
                         [[.02, .001, -.008], [.038, .001, .01]]])
    target = JawGeometry(surfaces, surfaces.reshape(-1, 3), directions, np.array([.044,.044]), ())
    source = JawGeometry(surfaces - [.002,0,0], surfaces.reshape(-1, 3),
                         directions, np.array([.045,.045]), ())
    return FirstLeftGrasp(None, source, target, [.03,.03,.1], pose_source="oracle_object_pose")


def test_centering_preserves_rotation_and_open_approach_distance():
    grasp = fixture()
    tool = np.eye(4)
    block = np.eye(4)
    block[:3,3] = [.085,.024,-.070]
    goal = grasp.center_target(tool,block)
    relative = inverse_pose(goal) @ block
    assert abs(relative[1,3]) < 1e-12
    np.testing.assert_allclose(relative[0,3] - grasp.target.center()[0],
                               .085 - grasp.source.center()[0], atol=1e-12)
    np.testing.assert_array_equal(tool[:3,:3],goal[:3,:3])
    assert goal[2,3] == 0.


def test_contact_gate_checks_both_sides_and_actual_depth():
    grasp = fixture()
    block = np.eye(4)
    block[:3,3] = [.029,0,-.070]
    assert grasp.open_clearance(np.eye(4),block,[1,1])["ready"]
    assert not grasp.open_clearance(np.eye(4),block,[.5,1])["ready"]
    prepared = grasp.center_target(np.eye(4),block,standoff=True)
    assert not grasp.open_clearance(prepared,block,[1,1])["ready"]
