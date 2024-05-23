"""
This function implements tensor completion using the Fully-Connected Tensor Netwrok (FCTN) 
method. 

This is a python implementation of the MatLab code in this repo: 
https://github.com/YuBangZheng/TenNet_ToolBox

The paper describing the algo is: 
https://cdn.aaai.org/ojs/17321/17321-13-20815-1-2-20210518.pdf
p11074
"""


import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao, mode_dot, tensordot


from tensorly.base import unfold, fold

def FCTN_TC(sparse_tensor, observed_entries_indices, max_R, rho=0.1, tol=1e-5, maxit=1000):

    # initialization begin
    N = sparse_tensor.ndim
    Nway = sparse_tensor.shape
    
    if max_R.shape[0] != max_R.shape[1]:
        raise ValueError('max_R should be an upper triangular square matrix.')

    X = sparse_tensor.copy()
    R = np.maximum(np.triu(np.ones((N, N)), 1) * 2, max_R - 5)
    print("DEBUG: R = \n ", R)
    
    tempdim = (np.diag(Nway) + R + R.T).astype(int)
    max_tempdim = np.diag(Nway) + max_R + max_R.T
    
    
    G = [np.random.normal(size = tempdim[i]) for i in range(N)]    
    r_change = 0.01

    for tmp in G: 
        print("DEBUG: tmp.shape = ", tmp.shape)


    # initialization end
    
    for k in range(maxit):
        Xold = X.copy()
        for i in range(N):
            Xi = unfold(X, mode=i)
            print("G[i].shape = ", G[i].shape)
            Gi = unfold(G[i], mode=i)
            print("Gi.shape = ", Gi.shape)
            print("sub_TN(G, i) = ", sub_TN(G,i))
            Girest = tnreshape(sub_TN(G, i), N, i)
            tempC = Xi @ Girest.T + rho * Gi
            tempA = Girest @ Girest.T + rho * np.eye(Gi.shape[1])
            G[i] = fold(tempC @ np.linalg.pinv(tempA), mode=i, shape=tempdim[i])
        
        X = (TN_composition(G) + rho * Xold) / (1 + rho)
        X[observed_entries_indices] = sparse_tensor[observed_entries_indices]
        
        rse = np.linalg.norm(X - Xold) / np.linalg.norm(Xold)
        
        if k % 10 == 0 or k == 1:
            print(f'FCTN-TC: iter = {k}   RSE = {rse}')
        
        if rse < tol:
            break
        
        rank_inc = (tempdim < max_tempdim).astype(int)
        if rse < r_change and np.sum(rank_inc) != 0:
            G = rank_inc_adaptive(G, rank_inc, N)
            tempdim += rank_inc
            r_change *= 0.5
    
    return X, G

def rank_inc_adaptive(G, rank_inc, N):
    for j in range(N):
        G[j] = np.pad(G[j], [(0, r) for r in rank_inc[j]], mode='constant', constant_values=np.random.rand())
    return G

def my_Unfold(tensor, shape, mode):
    return unfold(tensor, mode)

def my_Fold(matrix, shape, mode):
    return fold(matrix, mode, shape)

def tnreshape(tensor, N, mode):
    new_shape = list(tensor.shape)
    new_shape.pop(mode)
    return tensor.reshape(new_shape)


def sub_TN(G, k):
    #print("DEBUG: in sub_TN")
    N = len(G)
    #print("k = ", k)
    a_1 = list(range(k+1, N))
    a_2 = list(range(0, k+1))
    a = a_1 + a_2
    #print("DEBUG: a = ", a)
  
    
    #for i in list(range(1, k)) + list(range(k+1, N+1)):
    for i in [x for x in range(1, N) if x != k]:
        #print('i = ', i )
        #print("a = ", a)
        #print("pre transpose G[i].shape = ", G[i].shape)
        G[i] = np.transpose(G[i], a)  # Convert to zero-based indexing
        #print("post transpose G[i].shape = ", G[i].shape)
    
    m = [1]
    n = [0]
    Out = G[a[0]]
    M = N
    
    for i in range(1, N-1):
        """
        print("DEBUG: Out.shape = ", Out.shape)
        print("DEBUG: M = ", M)
        print("DEBUG: N = ", N)
        print("DEBUG: m = ", m)
        print("DEBUG: n = ", n)
        print("DEBUG: a[i] = ", a[i])
        print("DEBUG: G[a[i]].shape = ", G[a[i]].shape)
        """
        Out = tl.tenalg.tensordot(Out, G[a[i]], modes=[m,n])
        #Out = tensor_contraction(Out, G[a[i]], M, N, m, n)
        M = M + N - 2*i
        n.append(i)
        tempm = 1 + i * (N - i)
        if i > 1:
            m[1:] = [m[j] - (j + 1) for j in range(len(m) - 1)]
        m.append(tempm)

    #Now seems ok up to here
    
    p = [0] * (2 * (N - k - 1))
    for i in range(1, N - k ):
        p[2*i-2] = 2*i - 1
        p[2*i-1] = 2*i - 2
    """
    print("DEBUG: Out.shape = ", Out.shape)
    print("DEBUG: p = ", p)
    print("DEBUG: N = ", N)
    print("DEBUG: k = ", k)
    print("DEBUG: p + list(range(2 * (N - k), 2 * (N - 1))) = ", p + list(range(2 * (N - k), 2 * (N - 1))))
    """
    Out = np.transpose(Out, p + list(range(2 * (N - k), 2 * (N - 1))))
    #print("DEBUG: After first transpose: Out.shape = ", Out.shape)
    #print("DEBUG: list(range(2 * (N - k), 2 * (N - 1))) + list(range(2 * (N - k))) = \n ", list(range(2 * (N - k), 2 * (N - 1))) + list(range(2 * (N - k - 1))))
    Out = np.transpose(Out, list(range(2 * (N - k), 2 * (N - 1))) + list(range(2 * (N - k - 1))))
    
    return Out

def tensor_contraction(X, Y, Sx, Sy, n, m):
    Nx = X.ndim
    Ny = Y.ndim
    Lx = list(X.shape)
    Ly = list(Y.shape)
    
    if Nx < Sx:
        Lx.extend([1] * (Sx - Nx))
    if Ny < Sy:
        Ly.extend([1] * (Sy - Ny))
    
    indexx = list(range(Sx))
    indexy = list(range(Sy))
    
    for idx in sorted(n, reverse=True):
        del indexx[idx]
    for idx in sorted(m, reverse=True):
        del indexy[idx]
    
    tempX = np.transpose(X, indexx + n)
    tempXX = tempX.reshape(np.prod([Lx[i] for i in indexx]), np.prod([Lx[i] for i in n]))
    
    tempY = np.transpose(Y, m + indexy)
    tempYY = tempY.reshape(np.prod([Ly[i] for i in m]), np.prod([Ly[i] for i in indexy]))
    
    print("DEBUG: tempXX.shape = ", tempXX.shape)
    print("DEBUG: tempYY.shape = ", tempYY.shape)

    tempOut = np.dot(tempXX, tempYY)
    Out = tempOut.reshape([Lx[i] for i in indexx] + [Ly[i] for i in indexy])
    
    return Out


def TN_composition(G):
    # This function should compose the tensor from the factors
    # The specific implementation will depend on the exact method
    pass

# Example usage
# sparse_tensor = np.random.rand(4, 4, 4)  # Example tensor
# observed_entries_indices = np.array([[0, 1], [1, 2], [2, 3]])  # Example observed entries
# max_R = np.random.rand(4, 4)  # Example FCTN rank matrix
# X, G = FCTN_TC(sparse_tensor, observed_entries_indices, max_R)
