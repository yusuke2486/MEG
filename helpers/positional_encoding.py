import torch
import scipy.sparse as sp
import numpy as np

def positional_encoding(adj, pos_enc_dim):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    """
    N = adj.shape[0]
    A = sp.csr_matrix(adj)  # Convert to sparse matrix format for efficiency
    D = sp.diags(np.array(A.sum(axis=1)).flatten(), 0)
    L = D - A
    EigVal, EigVec = np.linalg.eigh(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVec = EigVec[:, idx]
    pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return pos_enc