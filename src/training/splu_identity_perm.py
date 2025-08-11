import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import splu, spilu
import os

# Load sparsity pattern
np.set_printoptions(threshold=np.inf)
DATA_DIR = "./BE_DATA/h2_10/"
NUM_SAMPLES = 898
sp_perm = np.load(DATA_DIR + "input_sparsity.npy")
n = sp_perm.shape[0]

''' BEGIN: REPLACE A BELOW WITH ACTUAL MATRIX '''
#load data
skipped = 0
X_list, y_list = [], []
for i in range(NUM_SAMPLES):
    A = np.loadtxt(
        os.path.join(DATA_DIR, f"jacobian_{i}.csv"), 
        delimiter=",", 
        dtype=np.float64
        )
    
    A[sp_perm == 0] = 0

    # Create triplets
    row, col = np.nonzero(sp_perm)
    data = A[row, col]

    # Create sparse matrix
    A_sparse = sparse.csc_array((data, (row, col)), shape=(n, n), dtype=np.float64)

    # Perform sparse LU without any permutations
    options = {}
    options["Equil"] = False # default True
    options["RowPerm"] = "NOROWPERM"
    options["SymmetricMode"] = False

    LU = splu(A_sparse, permc_spec="NATURAL", diag_pivot_thresh=0., options=options)

    # Check that splu permutation is identity
    assert(np.all(LU.perm_c == LU.perm_r))
    assert(np.all(LU.perm_c == np.arange(n)))

    # Check that LU reproduces A
    LU_product = LU.L @ LU.U
    assert(np.linalg.norm((A_sparse - LU_product).toarray()) < 1e-7)