import numpy as np

from geometry import closure_rows, intersects_box, relative
from input_semantics import substantial_red_regions, visible_silhouette


def test_surface_intersection_for_triangle_with_all_vertices_outside():
    triangle = np.array([[[-2., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
    assert intersects_box(triangle, np.ones(3))
    assert not intersects_box(triangle + [0., 0., 2.], np.ones(3))


def test_signed_jaw_lateral_separate_from_task_position():
    tool = np.eye(4)
    tool[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    block = np.eye(4)
    block[:3, 3] = [-.026, 0, 0]
    jaw = np.array([[.03, 0, 0]])
    np.testing.assert_array_equal(relative([tool], block, jaw)["block_xyz_jaw_mm"], [[-30., 26., 0.]])


def test_near_open_intent_is_not_effective_closure():
    chunk = np.ones((50, 14))
    chunk[48:,6] = [.94,.84]
    assert closure_rows(chunk) == {"0.95":49,"0.5":None,"0.1":None}


def test_clip_silhouette_before_taking_extrema():
    # Unclipped x=0 only occurs below the image; clamping the extrema invents
    # visible block coverage on the left.
    projected = np.array([[0,15],[15,0],[15,15]])
    clipped = visible_silhouette(projected,10,10)
    np.testing.assert_allclose(clipped.min(axis=0),[6,6])
    np.testing.assert_allclose(clipped.max(axis=0),[9,9])


def test_ignore_disconnected_red_background_speckle():
    image = np.zeros((3, 30, 10), dtype=np.uint8)
    image[0, 0, 0] = 200
    image[0, 4:10, 3:9] = 200
    image[0, 15:21, 3:9] = 200
    assert substantial_red_regions(image) == ([3,4,8,20],72)
