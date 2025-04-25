import numpy as np

# Load the .npy file
matrix_profile = np.load("utils/matrix_profile.npy", allow_pickle=True)
matrix_profile = np.array(matrix_profile, dtype=np.float64)

# Define a custom format string for aligned columns
fmt = "%14.6f %20.0f %24.0f %26.0f"

# Save to text file with proper formatting and header
np.savetxt(
    "utils/matrix_profile.txt",
    matrix_profile,
    fmt=fmt,
    header="   MatrixProfile     MatrixProfileIndex     LeftMatrixProfileIndex     RightMatrixProfileIndex",
    comments=""
)
