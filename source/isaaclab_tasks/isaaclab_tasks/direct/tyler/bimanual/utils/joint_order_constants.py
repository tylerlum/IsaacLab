from typing import List

import numpy as np
import torch

# [LEFT, RIGHT, LEFT, RIGHT, ...]
ISAACLAB_JOINT_ORDER = [
    "left_iiwa14_joint_1",
    "right_iiwa14_joint_1",
    "left_iiwa14_joint_2",
    "right_iiwa14_joint_2",
    "left_iiwa14_joint_3",
    "right_iiwa14_joint_3",
    "left_iiwa14_joint_4",
    "right_iiwa14_joint_4",
    "left_iiwa14_joint_5",
    "right_iiwa14_joint_5",
    "left_iiwa14_joint_6",
    "right_iiwa14_joint_6",
    "left_iiwa14_joint_7",
    "right_iiwa14_joint_7",
    "left_index_joint_0",
    "left_middle_joint_0",
    "left_ring_joint_0",
    "left_thumb_joint_0",
    "right_index_joint_0",
    "right_middle_joint_0",
    "right_ring_joint_0",
    "right_thumb_joint_0",
    "left_index_joint_1",
    "left_middle_joint_1",
    "left_ring_joint_1",
    "left_thumb_joint_1",
    "right_index_joint_1",
    "right_middle_joint_1",
    "right_ring_joint_1",
    "right_thumb_joint_1",
    "left_index_joint_2",
    "left_middle_joint_2",
    "left_ring_joint_2",
    "left_thumb_joint_2",
    "right_index_joint_2",
    "right_middle_joint_2",
    "right_ring_joint_2",
    "right_thumb_joint_2",
    "left_index_joint_3",
    "left_middle_joint_3",
    "left_ring_joint_3",
    "left_thumb_joint_3",
    "right_index_joint_3",
    "right_middle_joint_3",
    "right_ring_joint_3",
    "right_thumb_joint_3",
]


# [RIGHT_ARM, RIGHT_HAND, LEFT_ARM, LEFT_HAND]
FABRIC_JOINT_ORDER = [
    "right_iiwa14_joint_1",
    "right_iiwa14_joint_2",
    "right_iiwa14_joint_3",
    "right_iiwa14_joint_4",
    "right_iiwa14_joint_5",
    "right_iiwa14_joint_6",
    "right_iiwa14_joint_7",
    "right_index_joint_0",
    "right_index_joint_1",
    "right_index_joint_2",
    "right_index_joint_3",
    "right_middle_joint_0",
    "right_middle_joint_1",
    "right_middle_joint_2",
    "right_middle_joint_3",
    "right_ring_joint_0",
    "right_ring_joint_1",
    "right_ring_joint_2",
    "right_ring_joint_3",
    "right_thumb_joint_0",
    "right_thumb_joint_1",
    "right_thumb_joint_2",
    "right_thumb_joint_3",
    "left_iiwa14_joint_1",
    "left_iiwa14_joint_2",
    "left_iiwa14_joint_3",
    "left_iiwa14_joint_4",
    "left_iiwa14_joint_5",
    "left_iiwa14_joint_6",
    "left_iiwa14_joint_7",
    "left_index_joint_0",
    "left_index_joint_1",
    "left_index_joint_2",
    "left_index_joint_3",
    "left_middle_joint_0",
    "left_middle_joint_1",
    "left_middle_joint_2",
    "left_middle_joint_3",
    "left_ring_joint_0",
    "left_ring_joint_1",
    "left_ring_joint_2",
    "left_ring_joint_3",
    "left_thumb_joint_0",
    "left_thumb_joint_1",
    "left_thumb_joint_2",
    "left_thumb_joint_3",
]

# [RIGHT_ARM, LEFT_ARM, RIGHT_HAND, LEFT_HAND]
CUROBO_JOINT_ORDER = [
    "right_iiwa14_joint_1",
    "right_iiwa14_joint_2",
    "right_iiwa14_joint_3",
    "right_iiwa14_joint_4",
    "right_iiwa14_joint_5",
    "right_iiwa14_joint_6",
    "right_iiwa14_joint_7",
    "left_iiwa14_joint_1",
    "left_iiwa14_joint_2",
    "left_iiwa14_joint_3",
    "left_iiwa14_joint_4",
    "left_iiwa14_joint_5",
    "left_iiwa14_joint_6",
    "left_iiwa14_joint_7",
    "right_index_joint_0",
    "right_index_joint_1",
    "right_index_joint_2",
    "right_index_joint_3",
    "right_middle_joint_0",
    "right_middle_joint_1",
    "right_middle_joint_2",
    "right_middle_joint_3",
    "right_ring_joint_0",
    "right_ring_joint_1",
    "right_ring_joint_2",
    "right_ring_joint_3",
    "right_thumb_joint_0",
    "right_thumb_joint_1",
    "right_thumb_joint_2",
    "right_thumb_joint_3",
    "left_index_joint_0",
    "left_index_joint_1",
    "left_index_joint_2",
    "left_index_joint_3",
    "left_middle_joint_0",
    "left_middle_joint_1",
    "left_middle_joint_2",
    "left_middle_joint_3",
    "left_ring_joint_0",
    "left_ring_joint_1",
    "left_ring_joint_2",
    "left_ring_joint_3",
    "left_thumb_joint_0",
    "left_thumb_joint_1",
    "left_thumb_joint_2",
    "left_thumb_joint_3",
]

# [RIGHT_ARM, RIGHT_HAND, LEFT_ARM, LEFT_HAND]
PYBULLET_JOINT_ORDER = [
    "right_iiwa14_joint_1",
    "right_iiwa14_joint_2",
    "right_iiwa14_joint_3",
    "right_iiwa14_joint_4",
    "right_iiwa14_joint_5",
    "right_iiwa14_joint_6",
    "right_iiwa14_joint_7",
    "right_index_joint_0",
    "right_index_joint_1",
    "right_index_joint_2",
    "right_index_joint_3",
    "right_middle_joint_0",
    "right_middle_joint_1",
    "right_middle_joint_2",
    "right_middle_joint_3",
    "right_ring_joint_0",
    "right_ring_joint_1",
    "right_ring_joint_2",
    "right_ring_joint_3",
    "right_thumb_joint_0",
    "right_thumb_joint_1",
    "right_thumb_joint_2",
    "right_thumb_joint_3",
    "left_iiwa14_joint_1",
    "left_iiwa14_joint_2",
    "left_iiwa14_joint_3",
    "left_iiwa14_joint_4",
    "left_iiwa14_joint_5",
    "left_iiwa14_joint_6",
    "left_iiwa14_joint_7",
    "left_index_joint_0",
    "left_index_joint_1",
    "left_index_joint_2",
    "left_index_joint_3",
    "left_middle_joint_0",
    "left_middle_joint_1",
    "left_middle_joint_2",
    "left_middle_joint_3",
    "left_ring_joint_0",
    "left_ring_joint_1",
    "left_ring_joint_2",
    "left_ring_joint_3",
    "left_thumb_joint_0",
    "left_thumb_joint_1",
    "left_thumb_joint_2",
    "left_thumb_joint_3",
]

# [RIGHT_ARM, RIGHT_HAND, LEFT_ARM, LEFT_HAND]
VISER_JOINT_ORDER = [
    "right_iiwa14_joint_1",
    "right_iiwa14_joint_2",
    "right_iiwa14_joint_3",
    "right_iiwa14_joint_4",
    "right_iiwa14_joint_5",
    "right_iiwa14_joint_6",
    "right_iiwa14_joint_7",
    "right_index_joint_0",
    "right_index_joint_1",
    "right_index_joint_2",
    "right_index_joint_3",
    "right_middle_joint_0",
    "right_middle_joint_1",
    "right_middle_joint_2",
    "right_middle_joint_3",
    "right_ring_joint_0",
    "right_ring_joint_1",
    "right_ring_joint_2",
    "right_ring_joint_3",
    "right_thumb_joint_0",
    "right_thumb_joint_1",
    "right_thumb_joint_2",
    "right_thumb_joint_3",
    "left_iiwa14_joint_1",
    "left_iiwa14_joint_2",
    "left_iiwa14_joint_3",
    "left_iiwa14_joint_4",
    "left_iiwa14_joint_5",
    "left_iiwa14_joint_6",
    "left_iiwa14_joint_7",
    "left_index_joint_0",
    "left_index_joint_1",
    "left_index_joint_2",
    "left_index_joint_3",
    "left_middle_joint_0",
    "left_middle_joint_1",
    "left_middle_joint_2",
    "left_middle_joint_3",
    "left_ring_joint_0",
    "left_ring_joint_1",
    "left_ring_joint_2",
    "left_ring_joint_3",
    "left_thumb_joint_0",
    "left_thumb_joint_1",
    "left_thumb_joint_2",
    "left_thumb_joint_3",
]

assert (
    len(CUROBO_JOINT_ORDER)
    == len(PYBULLET_JOINT_ORDER)
    == len(ISAACLAB_JOINT_ORDER)
    == len(FABRIC_JOINT_ORDER)
    == len(VISER_JOINT_ORDER)
), (
    f"{len(CUROBO_JOINT_ORDER)}, {len(PYBULLET_JOINT_ORDER)}, {len(ISAACLAB_JOINT_ORDER)}, {len(FABRIC_JOINT_ORDER)}, {len(VISER_JOINT_ORDER)}"
)
assert (
    set(CUROBO_JOINT_ORDER)
    == set(PYBULLET_JOINT_ORDER)
    == set(ISAACLAB_JOINT_ORDER)
    == set(FABRIC_JOINT_ORDER)
    == set(VISER_JOINT_ORDER)
), (
    f"{set(CUROBO_JOINT_ORDER)}, {set(PYBULLET_JOINT_ORDER)}, {set(ISAACLAB_JOINT_ORDER)}, {set(FABRIC_JOINT_ORDER)}, {set(VISER_JOINT_ORDER)}"
)


def change_joint_order(
    q: np.ndarray, from_order: List[str], to_order: List[str]
) -> np.ndarray:
    assert q.ndim in [1, 2], f"q.ndim: {q.ndim}"
    original_shape = q.shape
    if q.ndim == 1:
        q = q[None]

    assert q.ndim == 2, f"q.shape: {q.shape}"
    N, D = q.shape
    assert D == len(from_order), f"D: {D} != len(from_order): {len(from_order)}"

    # q is given as from_order
    # Thus for each idx in q, we can map that joint from_order[idx] to that joint value q[idx]
    joint_name_to_value = {from_order[i]: q[:, i] for i in range(D)}
    new_q = np.stack(
        [joint_name_to_value[joint_name] for joint_name in to_order], axis=1
    )
    assert new_q.shape == (N, len(to_order)), (
        f"new_q.shape: {new_q.shape}, len(to_order): {len(to_order)}"
    )
    return new_q.reshape(original_shape)


def change_joint_order_torch(
    q: torch.Tensor, from_order: List[str], to_order: List[str]
) -> torch.Tensor:
    assert q.ndim in [1, 2], f"q.ndim: {q.ndim}"
    original_shape = q.shape
    if q.ndim == 1:
        q = q[None]

    assert q.ndim == 2, f"q.shape: {q.shape}"
    N, D = q.shape
    assert D == len(from_order), f"D: {D} != len(from_order): {len(from_order)}"
    joint_name_to_value = {from_order[i]: q[:, i] for i in range(D)}
    new_q = torch.stack(
        [joint_name_to_value[joint_name] for joint_name in to_order], dim=1
    )
    assert new_q.shape == (N, len(to_order)), (
        f"new_q.shape: {new_q.shape}, len(to_order): {len(to_order)}"
    )
    return new_q.reshape(original_shape)


def pybullet_to_isaaclab_joint_order(q: np.ndarray) -> np.ndarray:
    return change_joint_order(
        q, from_order=PYBULLET_JOINT_ORDER, to_order=ISAACLAB_JOINT_ORDER
    )


def isaaclab_to_pybullet_joint_order(q: np.ndarray) -> np.ndarray:
    return change_joint_order(
        q, from_order=ISAACLAB_JOINT_ORDER, to_order=PYBULLET_JOINT_ORDER
    )


def isaaclab_to_curobo_joint_order(q: np.ndarray) -> np.ndarray:
    return change_joint_order(
        q, from_order=ISAACLAB_JOINT_ORDER, to_order=CUROBO_JOINT_ORDER
    )


def curobo_to_isaaclab_joint_order(q: np.ndarray) -> np.ndarray:
    return change_joint_order(
        q, from_order=CUROBO_JOINT_ORDER, to_order=ISAACLAB_JOINT_ORDER
    )


def fabric_to_isaaclab_joint_order(q: np.ndarray) -> np.ndarray:
    return change_joint_order(
        q, from_order=FABRIC_JOINT_ORDER, to_order=ISAACLAB_JOINT_ORDER
    )


def isaaclab_to_fabric_joint_order(q: np.ndarray) -> np.ndarray:
    return change_joint_order(
        q, from_order=ISAACLAB_JOINT_ORDER, to_order=FABRIC_JOINT_ORDER
    )


def pybullet_to_isaaclab_joint_order_torch(q: torch.Tensor) -> torch.Tensor:
    return change_joint_order_torch(
        q, from_order=PYBULLET_JOINT_ORDER, to_order=ISAACLAB_JOINT_ORDER
    )


def isaaclab_to_pybullet_joint_order_torch(q: torch.Tensor) -> torch.Tensor:
    return change_joint_order_torch(
        q, from_order=ISAACLAB_JOINT_ORDER, to_order=PYBULLET_JOINT_ORDER
    )


def isaaclab_to_curobo_joint_order_torch(q: torch.Tensor) -> torch.Tensor:
    return change_joint_order_torch(
        q, from_order=ISAACLAB_JOINT_ORDER, to_order=CUROBO_JOINT_ORDER
    )


def curobo_to_isaaclab_joint_order_torch(q: torch.Tensor) -> torch.Tensor:
    return change_joint_order_torch(
        q, from_order=CUROBO_JOINT_ORDER, to_order=ISAACLAB_JOINT_ORDER
    )


def fabric_to_isaaclab_joint_order_torch(q: torch.Tensor) -> torch.Tensor:
    return change_joint_order_torch(
        q, from_order=FABRIC_JOINT_ORDER, to_order=ISAACLAB_JOINT_ORDER
    )


def isaaclab_to_fabric_joint_order_torch(q: torch.Tensor) -> torch.Tensor:
    return change_joint_order_torch(
        q, from_order=ISAACLAB_JOINT_ORDER, to_order=FABRIC_JOINT_ORDER
    )
