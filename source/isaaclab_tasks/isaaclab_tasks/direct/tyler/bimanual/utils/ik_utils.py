import torch


def control_ik(
    j_eef: torch.Tensor, dpose: torch.Tensor, damping: float = 0.5
) -> torch.Tensor:
    # solve damped least squares

    # Set controller parameters
    NUM_ENVS, NUM_EE_DIMS, NUM_JOINTS = j_eef.shape
    assert dpose.shape == (NUM_ENVS, NUM_EE_DIMS), f"dpose.shape: {dpose.shape}"

    j_eef_T = torch.transpose(j_eef, 1, 2)
    assert j_eef_T.shape == (NUM_ENVS, NUM_JOINTS, NUM_EE_DIMS), (
        f"j_eef_T.shape: {j_eef_T.shape}"
    )

    lmbda = torch.eye(NUM_EE_DIMS, device=j_eef.device) * (damping**2)
    assert lmbda.shape == (NUM_EE_DIMS, NUM_EE_DIMS), f"lmbda.shape: {lmbda.shape}"

    # u = J.T @ (J @ J.T + lambda)^-1 @ dpose
    u = torch.bmm(
        j_eef_T,
        torch.bmm(
            torch.inverse(torch.bmm(j_eef, j_eef_T) + lmbda[None]),
            dpose.unsqueeze(dim=-1),
        ),
    ).squeeze(dim=-1)
    assert u.shape == (NUM_ENVS, NUM_JOINTS), f"u.shape: {u.shape}"

    return u
