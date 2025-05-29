import torch
from isaaclab_tasks.direct.tyler.bimanual.utils.constants import NUM_QUAT, NUM_XYZ
from isaaclab_tasks.direct.tyler.bimanual.utils.torch_utils import quat_rotate


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


NUM_OBJECT_KEYPOINTS = 3

OBJECT_KEYPOINTS_LEN = 0.2

OBJECT_KEYPOINT_OFFSETS = [
    [OBJECT_KEYPOINTS_LEN, 0.0, 0.0],
    [0.0, OBJECT_KEYPOINTS_LEN, 0.0],
    [0.0, 0.0, OBJECT_KEYPOINTS_LEN],
]
# Useful for objects like the cup or plate that are symmetric around the z-axis
OBJECT_KEYPOINT_OFFSETS_YAW_INVARIANT = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, OBJECT_KEYPOINTS_LEN],
]

OBJECT_KEYPOINT_OFFSETS_ROT_INVARIANT = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]
assert (
    len(OBJECT_KEYPOINT_OFFSETS)
    == len(OBJECT_KEYPOINT_OFFSETS_ROT_INVARIANT)
    == NUM_OBJECT_KEYPOINTS
)


def compute_keypoint_positions(
    pos: torch.Tensor,
    quat_xyzw: torch.Tensor,
    keypoint_offsets: torch.Tensor,
) -> torch.Tensor:
    N, _ = pos.shape
    assert_equals(pos.shape, (N, NUM_XYZ))
    assert_equals(quat_xyzw.shape, (N, NUM_QUAT))
    n_keypoints = keypoint_offsets.shape[1]
    assert_equals(keypoint_offsets.shape, (N, n_keypoints, NUM_XYZ))

    # Rotate keypoint offsets by quat_xyzw
    keypoint_offsets_rotated = torch.zeros_like(
        keypoint_offsets, device=keypoint_offsets.device
    )
    for i in range(n_keypoints):
        keypoint_offsets_i = keypoint_offsets[:, i]
        assert_equals(keypoint_offsets_i.shape, (N, NUM_XYZ))
        keypoint_offsets_rotated_i = quat_rotate(q=quat_xyzw, v=keypoint_offsets_i)
        assert_equals(keypoint_offsets_rotated_i.shape, (N, NUM_XYZ))

        keypoint_offsets_rotated[:, i] = keypoint_offsets_rotated_i

    # Add to pos
    keypoint_positions = pos.unsqueeze(dim=1) + keypoint_offsets_rotated
    assert_equals(keypoint_positions.shape, (N, n_keypoints, NUM_XYZ))
    return keypoint_positions
