import numpy as np
from scipy.linalg import lu

# example matrix
A = np.array([[4, 3, 2],
              [6, 3, 0],
              [2, 1, 1]], float)

# do an LU factorization (with no pivoting, for simplicity)
# if you’re using your own lu_decompose routine just replace this
P, L, U = lu(A)  

# pack L and U into one matrix:
#   • below the diagonal (i>j) take L[i,j]
#   • on and above the diagonal (i<=j) take U[i,j]
LU_packed = np.tril(L, -1) + U

print("L:\n", L)
print("U:\n", U)
print("packed LU:\n", LU_packed)

# now flatten:
y = LU_packed.ravel()   # or .flatten()
print("flattened:", y)
