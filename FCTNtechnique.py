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
    
    tempdim = (np.diag(Nway) + R + R.T).astype(int)
    max_tempdim = np.diag(Nway) + max_R + max_R.T
    
    
    G = [np.random.normal(size = tempdim[i]) for i in range(N)]    
    r_change = 0.01


    # initialization end
    
    for k in range(maxit):
        Xold = X.copy()
        for i in range(N):
            print("___________________________________")
            Xi = unfold(X, mode=i)
            Gi = unfold(G[i], mode=i)
            #print("DEBUG: i = ", i)
            #print("DEBUG: G[i].shape = ", G[i].shape)
            #print("DEBUG: G = ")
            #for g in G: 
            #    print("DEBUG: g.shape = ", g.shape)
            
            Girest = sub_TN(G, i)
            Girest = tnreshape(Girest, N, i)
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

"""
def tnreshape(tensor, N, mode):
    new_shape = list(tensor.shape)
    new_shape.pop(mode)
    return tensor.reshape(new_shape)
"""
def tnreshape(Grest, N, i):
  """
  Reshapes a tensor according to Theorem 4 in [1].

  Args:
      Grest: A tensor of size n_1 * R_{1,k} * ... * n_{k-1} * R_{k-1,k} * R_{k,k+1} * n_{k+1} * ... * R_{k,N} * n_N.
      N: The dimension of tensor X.
      i: The ith dimension.

  Returns:
      Out: A reshaped tensor of size (R_{1,k}...R_{k-1,k}R_{k,k+1}...R{k,N}) * (n_1...n_{k-1}n_{k+1}...n_{N}).
  """

  # Handle cases where Grest has fewer dimensions than expected
  Nway = np.array(Grest.shape)
  #print("DEBUG: pre loop Nway = ", Nway)
  if len(Nway) < 2 * (N - 1):
    Nway = np.concatenate((Nway, np.ones(2 * (N - 1) - len(Nway))))
  #print("DEBUG: post loop Nway = ", Nway)
  # Create arrays to store reshaped dimensions
  m = np.zeros(N - 1, dtype=int)
  n = np.zeros(N - 1, dtype=int)

  # Define permutation based on dimension i
  #print("DEBUG: i = ", i)
  for k in range(N - 1):
    #print("DEBUG: k = ", k)
    if k < i:
      #print("in if")
      m[k] = 2 * (k+1) - 1
      n[k] = 2 * (k+1) - 2
    else:
      #print("in else")
      m[k] = 2 * (k+1) - 2
      n[k] = 2 * (k+1) - 1
    #print("m[k] = ", m[k])
    #print("n[k] = ", n[k])

  # Permute the tensor based on the calculated indices
  #print("DEBUG: Grest.shape = ", Grest.shape)
  #print("DEBUG: m = ", m)
  #print("DEBUG: n = ", n)
  dimorder =  np.concatenate((m,n)) #N.B. original matlab code was passing in a 2D array but think that was error (changed it in matlab and didn't seem to break anything)
  tempG = np.transpose(Grest, dimorder)
  """ 
  print("DEBUG: tempG.shape = ", tempG.shape)
  print("DEBUG: Nway.shape = ", Nway.shape)
  print("DEBUG: m = ", m)
  print("DEBUG: n = ", n)
  print("DEBUG: Nway[m] = ", Nway[m])
  print("DEBUG: Nway[n] = ", Nway[n])
  print("DEBUG: np.prod(Nway[m]) = ", np.prod(Nway[m]))
  print("DEBUG: np.prod(Nway[n]) = ", np.prod(Nway[n]))
  """
  
  # Reshape the permuted tensor
  Out = tempG.reshape((np.prod(Nway[m]), np.prod(Nway[n])))

  return Out


def sub_TN(G, k):
    #print("DEBUG: in sub_TN")
    G_copy = G.copy()
    #print("DEBUG: at input of sub_TN shape of G_copy:")
    #for g_copy in G_copy:
    #    print("DEBUG: g_copy.shape = ", g_copy.shape)
    N = len(G_copy)
    #print("k = ", k)
    a_1 = list(range(k+1, N))
    a_2 = list(range(0, k+1))
    a = a_1 + a_2
    print("DEBUG: a = ", a)
  
    #print("DEBUG: pre transpose: G_copy[3].shape = ", G_copy[3].shape)
    #for i in list(range(1, k)) + list(range(k+1, N+1)):
    for i in [x for x in range(N) if x != k]:
        #print('i = ', i )
        #print("a = ", a)
        #print("pre transpose G_copy[i].shape = ", G_copy[i].shape)
        G_copy[i] = np.transpose(G_copy[i], a)  # Convert to zero-based indexing
        #print("post transpose G_copy[i].shape = ", G_copy[i].shape)
    #print("DEBUG: post transpose: G_copy[3].shape = ", G_copy[3].shape)
    print("DEBUG: after transposing shape of G_copy:")
    #for g_copy in G_copy:
    #    print("DEBUG: g_copy.shape = ", g_copy.shape)
    m = [1]
    n = [0]
    Out = G_copy[a[0]]
    M = N
    
    for i in range(1, N-1):
        print("========")
        """
        print("DEBUG: Out.shape = ", Out.shape)
        print("DEBUG: M = ", M)
        print("DEBUG: N = ", N)
        print("DEBUG: m = ", m)
        print("DEBUG: n = ", n)
        print("DEBUG: a[i] = ", a[i])
        """
        #print("DEBUG: G_copy[a[i]].shape = ", G_copy[a[i]].shape)

        #print("DEBUG: shape of G_copy:")
        #for g_copy in G_copy:
        #    print("DEBUG: g_copy.shape = ", g_copy.shape)
            
        
        
        Out = tl.tenalg.tensordot(Out, G_copy[a[i]], modes=[m,n])
        #Out = tensor_contraction(Out, G_copy[a[i]], M, N, m, n)
        #print("DEBUG: NEW Out.shape = ", Out.shape)
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
    
    #print("DEBUG: Out.shape = ", Out.shape)
    #print("DEBUG: p = ", p)
    #print("DEBUG: N = ", N)
    #print("DEBUG: k = ", k)
    #print("DEBUG: p = ", p)
    extra_p_oneindex = list(range(2 * (N - (k+1)) + 1, 2 * (N - 1) + 1 ))
    extra_p = [x - 1 for x in extra_p_oneindex] # now express in zero indexed terms
    
    Out = np.transpose(Out, p + extra_p)
    print("DEBUG: After first transpose: Out.shape = ", Out.shape)
    #print("DEBUG: list(range(2 * (N - (k+1)) + 1, 2 * (N - 1) + 1))  = \n ",  )
    #print("DEBUG: list(range(1, 2 * (N - (k+1)) + 1)) = \n ", ))
    #print("DEBUG: list(range(2 * (N - k), 2 * (N - 1))) + list(range(2 * (N - k))) = \n ", list(range(2 * (N - (k+1)) + 1, 2 * (N - 1) + 1)) + list(range(1, 2 * (N - (k+1) + 1))))
    second_p_oneindex = list(range(2 * (N - (k+1)) + 1, 2 * (N - 1) + 1)) + list(range(1, 2 * (N - (k+1)) + 1))
    second_p = [x - 1 for x in second_p_oneindex] # now express in zero indexed terms
    print("DEBUG: second_p = ", second_p)
    Out = np.transpose(Out, second_p)
    
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
