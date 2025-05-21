import numpy as np
from scipy.spatial.transform import Rotation as R

TABLE_LENGTH_X, TABLE_LENGTH_Y, TABLE_LENGTH_Z = (
    0.76,
    1.53,
    0.05,
)  # BRITTLE: Must match table.urdf

# TC is table center
X_R_TC = np.eye(4)
X_R_TC[:3, 3] = np.array(
    [
        TABLE_LENGTH_X / 2,
        0,
        0,
    ]
)

TABLE_X, TABLE_Y, TABLE_Z = X_R_TC[:3, 3]
TABLE_QX, TABLE_QY, TABLE_QZ, TABLE_QW = R.from_matrix(X_R_TC[:3, :3]).as_quat()
