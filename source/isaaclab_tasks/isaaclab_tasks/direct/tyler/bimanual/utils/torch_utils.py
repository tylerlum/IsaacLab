import torch
import torch.nn.functional as F


def assert_equals(a, b):
    assert a == b, f"a: {a}, b: {b}"


def sample_uniform_tensor(
    low: torch.Tensor, high: torch.Tensor, N: int
) -> torch.Tensor:
    assert low.ndim == high.ndim == 1, f"low.ndim: {low.ndim}, high.ndim: {high.ndim}"
    D = low.shape[0]
    assert low.shape == high.shape == (D,), (
        f"low.shape: {low.shape}, high.shape: {high.shape}, D: {D}"
    )
    return low + (high - low) * torch.rand(N, D, device=low.device)


def quat_wxyz_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quat_wxyz: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quat_wxyz, -1)
    two_s = 2.0 / (quat_wxyz * quat_wxyz).sum(-1)

    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return mat.reshape(quat_wxyz.shape[:-1] + (3, 3))


def quat_xyzw_to_matrix(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return quat_wxyz_to_matrix(quat_wxyz)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    subgradient is zero where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def matrix_to_quat_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with imaginary part first, as tensor of shape (..., 4).
    """
    quat_wxyz = matrix_to_quat_wxyz(matrix)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return quat_xyzw


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            torch.clamp(matrix[..., i0, i2], -1.0 + 1e-8, 1.0 - 1e-8)
            * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(
            torch.clamp(matrix[..., i0, i0], -1.0 + 1e-8, 1.0 - 1e-8)
        )

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


# ---------- 1‑D piece‑wise linear interpolation ----------
def interpolate(t: torch.Tensor, x: torch.Tensor, new_t: torch.Tensor) -> torch.Tensor:
    """
    t      : (N,)          – strictly‑increasing time‑stamps
    x      : (N, D)        – values at t
    new_t  : (M,)          – query times  (need not be sorted)

    Returns: (M, D)        – linearly‑interpolated values
    """
    assert t.ndim == 1 and x.ndim == 2 and new_t.ndim == 1, (
        f"Shapes – t:{t.shape}, x:{x.shape}, new_t:{new_t.shape}"
    )
    N = t.shape[0]
    _M, D = new_t.shape[0], x.shape[1]
    assert x.shape == (N, D), f"x.shape: {x.shape}, expected: {(N, D)}"

    # 1) clamp to support only in‑range queries
    new_t_clamped = new_t.clamp(min=t[0], max=t[-1])

    # 2) find the enclosing interval  (right=True  ⇒ idx‑1 ≤ v < idx)
    idx = torch.searchsorted(t, new_t_clamped, right=True)
    idx = idx.clamp(min=1, max=t.numel() - 1)

    # 3) gather endpoints
    t0, t1 = t[idx - 1], t[idx]  # (M,)
    x0, x1 = x[idx - 1], x[idx]  # (M, D)

    # 4) linear blend
    alpha = ((new_t_clamped - t0) / (t1 - t0)).unsqueeze(1)  # (M,1)
    return x0 + alpha * (x1 - x0)


# ---------- Quaternion SLERP -----------------------------------------------
def _vectorized_slerp(
    q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """
    q0, q1 : (B,4)  – unit quats in the same convention
    alpha  : (B,)   – interpolation factors in [0,1]
    """
    q0 = torch.nn.functional.normalize(q0, p=2, dim=1)
    q1 = torch.nn.functional.normalize(q1, p=2, dim=1)

    dots = (q0 * q1).sum(dim=1)  # (B,)

    # take the short path
    neg_mask = dots < 0
    q1 = torch.where(neg_mask.unsqueeze(1), -q1, q1)
    dots = torch.where(neg_mask, -dots, dots)

    close_mask = dots > 0.9995
    out = torch.empty_like(q0)

    # -- linear blend for nearly identical quats
    if close_mask.any():
        a = alpha[close_mask].unsqueeze(1)
        lerp = q0[close_mask] * (1.0 - a) + q1[close_mask] * a
        out[close_mask] = torch.nn.functional.normalize(lerp, p=2, dim=1)

    # -- true slerp elsewhere
    nc_mask = ~close_mask
    if nc_mask.any():
        dot_nc = dots[nc_mask]
        theta = torch.acos(torch.clamp(dot_nc, -1.0 + 1e-8, 1.0 - 1e-8))
        sin_theta = torch.sin(theta)
        a_nc = alpha[nc_mask]
        q0_nc, q1_nc = q0[nc_mask], q1[nc_mask]

        s0 = torch.sin((1.0 - a_nc) * theta) / sin_theta
        s1 = torch.sin(a_nc * theta) / sin_theta

        out_nc = q0_nc * s0.unsqueeze(1) + q1_nc * s1.unsqueeze(1)
        out[nc_mask] = torch.nn.functional.normalize(out_nc, p=2, dim=1)

    return out


def interpolate_quats(
    t: torch.Tensor, x: torch.Tensor, new_t: torch.Tensor
) -> torch.Tensor:
    """
    t     : (N,)
    x     : (N,4)  – quats
    new_t : (M,)
    Returns (M,4)
    """
    new_t_clamped = new_t.clamp(min=t[0], max=t[-1])
    idx = torch.searchsorted(t, new_t_clamped, right=True).clamp(1, t.numel() - 1)

    t0, t1 = t[idx - 1], t[idx]
    alpha = (new_t_clamped - t0) / (t1 - t0)  # (M,)

    q0, q1 = x[idx - 1], x[idx]  # (M,4)
    return _vectorized_slerp(q0, q1, alpha)


# ---------- Rotations & full 4×4 poses -------------------------------------
def interpolate_rotation(
    t: torch.Tensor, R: torch.Tensor, new_t: torch.Tensor
) -> torch.Tensor:
    """
    R : (N,3,3) rotation matrices
    Returns (M,3,3)
    """
    quats = matrix_to_quat_xyzw(R)  # (N,4)
    new_quats = interpolate_quats(t, quats, new_t)  # (M,4)
    return quat_xyzw_to_matrix(new_quats)  # (M,3,3)


def interpolate_poses(
    t: torch.Tensor, T: torch.Tensor, new_t: torch.Tensor
) -> torch.Tensor:
    """
    T : (N,4,4) SE(3) matrices
    Returns (M,4,4)
    """
    assert T.ndim == 3 and T.shape[1:] == (4, 4)
    device, dtype = T.device, T.dtype
    M = new_t.shape[0]

    out = torch.eye(4, device=device, dtype=dtype).repeat(M, 1, 1)
    # translation
    out[:, :3, 3] = interpolate(t, T[:, :3, 3], new_t)
    # rotation
    out[:, :3, :3] = interpolate_rotation(t, T[:, :3, :3], new_t)
    return out


def rescale(
    values: torch.Tensor,
    old_mins: torch.Tensor,
    old_maxs: torch.Tensor,
    new_mins: torch.Tensor,
    new_maxs: torch.Tensor,
):
    """
    Rescale the input tensor from the old range to the new range.

    Args:
    values (torch.Tensor): Input tensor to be rescaled, shape (N, M)
    old_mins (torch.Tensor): Minimum values of the old range, shape (M,)
    old_maxs (torch.Tensor): Maximum values of the old range, shape (M,)
    new_mins (torch.Tensor): Minimum values of the new range, shape (M,)
    new_maxs (torch.Tensor): Maximum values of the new range, shape (M,)

    Returns:
    torch.Tensor: Rescaled tensor, shape (N, M)
    """
    assert_equals(len(values.shape), 2)
    N, M = values.shape
    assert_equals(old_mins.shape, (M,))
    assert_equals(old_maxs.shape, (M,))
    assert_equals(new_mins.shape, (M,))
    assert_equals(new_maxs.shape, (M,))

    # Ensure all inputs are tensors and on the same device
    old_mins = torch.as_tensor(old_mins, dtype=values.dtype, device=values.device)
    old_maxs = torch.as_tensor(old_maxs, dtype=values.dtype, device=values.device)
    new_mins = torch.as_tensor(new_mins, dtype=values.dtype, device=values.device)
    new_maxs = torch.as_tensor(new_maxs, dtype=values.dtype, device=values.device)

    # Clip the input values to be within the old range
    values_clipped = torch.clamp(values, min=old_mins[None], max=old_maxs[None])

    # Perform the rescaling
    rescaled = (values_clipped - old_mins[None]) / (old_maxs[None] - old_mins[None]) * (
        new_maxs[None] - new_mins[None]
    ) + new_mins[None]

    return rescaled


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    N = T.shape[0]
    assert T.shape == (N, 4, 4), f"T.shape: {T.shape}, expected: {(N, 4, 4)}"
    assert points.shape == (N, 3), f"points.shape: {points.shape}, expected: {(N, 3)}"

    # (N,4,1): make each point homogeneous by appending 1 and adding a dummy axis
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1).unsqueeze(
        -1
    )

    # (N,4,1): batched transform
    points_tf = torch.bmm(T, points_h)

    # (N,3): discard the homogeneous component
    return points_tf[:, :3, 0]


def pose_to_T(pose: torch.Tensor) -> torch.Tensor:
    N = pose.shape[0]
    assert pose.shape == (N, 7), f"pose.shape: {pose.shape}, expected: {(N, 7)}"
    T = torch.eye(4, device=pose.device, dtype=pose.dtype).repeat(N, 1, 1)
    pos = pose[:, :3]
    quat_wxyz = pose[:, 3:]
    T[:, :3, 3] = pos
    T[:, :3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    return T
