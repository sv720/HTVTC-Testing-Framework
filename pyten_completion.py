import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)
sys.path.append('/home/scott/Imperial/Year4/FYP/pyten')

from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import copy
import classificationmetrics
from FCTNtechnique import FCTN_TC
import numpy as np
from crosstechnique import generateCrossComponents, noisyReconstruction
from sketchtechnique import tensorCompletionSketchingMRP
from generateerrortensor import generateIncompleteErrorTensor
import time
from tensorly.tenalg import multi_mode_dot
from tensorly import unfold
import pyten
import pandas as pd

def pyten_TC(sparse_tensor, function_name, r=20, tol=1e-4, maxiter=100, init='random', omega=None, recover=None, printitn=0):
    subs = list(np.ndindex(sparse_tensor.shape))
    vals = np.array([sparse_tensor[idx] for idx in subs])



    vals=vals.reshape(vals.shape[0], 1)

    # First: create Sptensor
    X1 = pyten.tenclass.Sptensor(subs=subs, vals=vals, shape=sparse_tensor.shape)
    # Second: create Tensor object and find missing data
    X = X1.totensor()
    Ori = X.data
    lstnan = np.isnan(X.data)
    X.data = np.nan_to_num(X.data)

    # Construct omega
    output = 1  # An output indicate flag. (Decompose: 1, Recover:2)
    if type(omega) != np.ndarray:
        # if True in lstnan:
        omega = X.data * 0 + 1
        omega[lstnan] = 0
        if recover == '1':
            output = 2

    # Choose method to recover or decompose
    if type(function_name) == str:
        if function_name == '1' or function_name == 'tucker_als':
            #print('DEBUG: X.shape = ', X.shape)
            #print('DEBUG: r = ', r)
            #print('DEBUG: omega.shape = ', omega.shape)
            #print('DEBUG: tol = ', tol)
            [Final, Rec] = pyten.method.tucker_als(X, r, omega, tol, maxiter, init, printitn)
            full = Final.totensor()
            Final_tensor_real = np.real(Final.totensor().data)
        elif function_name == '2' or function_name == 'cp_als':
            #print("DEBUG: X.shape = ", X.shape)
            #print("DEBUG: r = ", r)
            [Final, Rec] = pyten.method.cp_als(X, r, omega, tol, maxiter, init, printitn)
            full = Final.totensor()
            Final_tensor_real = np.real(Final.totensor().data)
        elif function_name == '3' or function_name == 'TNCP':
            Omega1 = pyten.tenclass.Tensor(omega)
            X_pyten_tenclass_Tensor = pyten.tenclass.Tensor(X.data)
            NNCP = pyten.method.TNCP(X_pyten_tenclass_Tensor, Omega1, r, tol, maxiter)
            NNCP.run()
            Final = NNCP.U
            Rec = NNCP.X
            full = NNCP.II.copy()
            for i in range(NNCP.ndims):
                full = full.ttm(NNCP.U[i], i + 1)
            Final_tensor_real = Rec.data
        elif function_name == '4' or function_name == 'SiLRTC':
            Rec = pyten.method.silrtc(X, omega, max_iter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif function_name == '5' or function_name == 'FaLRTC':
            Rec = pyten.method.falrtc(X, omega, max_iter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif function_name == '6' or function_name == 'HaLRTC':
            Rec = pyten.method.halrtc(X, omega, max_iter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif function_name == '7' or function_name == 'PARAFAC2':
            X1 = [X.data]
            multi = input("Please input how many other multiset files you want to couple with the first one "
                                "(Input 'None' if no other info.) \n")
            """
            if multi != 'None':
                for i in range(int(multi)):
                    FileName2 = input(
                        "Please input the file_name of the " + str(i + 2) + " slice of the multiset data:\n")
                    if FileName2 != 'None':
                        dat2 = pd.read_csv(FileName2, delimiter=';')
                        # Data preprocessing
                        # First: create Sptensor
                        dat2v = dat2.values
                        sha2v = dat2v.shape
                        subs2v = dat2v[:, range(sha2v[1] - 1)]
                        subs2v = subs2v - 1
                        vals2 = dat2v[:, sha2v[1] - 1]
                        vals2 = vals2.reshape(len(vals2), 1)
                        siz2 = np.max(subs2v, 0)
                        siz2 = np.int32(siz2 + 1)
                        X2 = pyten.tenclass.Sptensor(subs2v, vals2, siz2)

                        # Second: create Tensor object and find missing data
                        X2 = X2.totensor()
                        # lstnan = np.isnan(X2.data)
                        X2.data = np.nan_to_num(X2.data)
                        X1.append(X2.data)
                print(X1[0].shape)
                print(X1[1].shape)
                print(X1[2].shape)
                print("\n")
            """
            parafac = pyten.method.PARAFAC2(X1, r, maxiter=maxiter, printitn=printitn)
            parafac.run()
            Ori = parafac.X
            Final = parafac
            Rec = None
            full = parafac.fit
        elif function_name == '8' or function_name == 'DEDICOM':
            dedicom = pyten.method.DEDICOM(X, r, maxiter=maxiter, printitn=printitn)
            dedicom.run()
            Final = dedicom
            Rec = None
            full = dedicom.fit
        elif function_name == '0':
            print('Successfully Exit')
            #return None, None, None, None
        else:
            raise ValueError('No Such Method')

    else:
        raise TypeError('No Such Method')

    # Output Result
    #print("DEBUG: np.array(subs).shape = ", np.array(subs).shape)
    [nv, nd] = np.array(subs).shape
    if output == 1:
        if function_name == '7':
            newsubs = []
            # tempvals = []
            # for i in range(int(multi)+1):
            # temp = pyten.tenclass.Tensor(full[i])
            # newsubs = newsubs.append(temp.tosptensor().subs)
            # tempvals = tempvals.append(temp.tosptensor().vals)
            # newfilename = file_name[:-4] + '_Decomposite' + file_name[-4:]
            # print("\n" + "The original Multiset Data is: ")
            # print(Ori)
            # print("\n" + "The Decomposed Result is: ")
            # print(Final)
        else:
            newsubs = full.tosptensor().subs
            tempvals = full.tosptensor().vals
            # newfilename = file_name[:-4] + '_Decomposite' + file_name[-4:]
            #print("\n" + "The original Tensor is: ")
            #print(Ori)
            #print("\n" + "The Decomposed Result is: ")
            #print(Final)
            #print('DEBUG: type(Final.totensor()) = ', type(Final.totensor()))
            #print('DEBUG: Final.totensor())= ', Final.totensor())
            #print('DEBUG: Final.totensor().data) = \n  ', np.real(Final.totensor().data))
            
    else:
        newsubs = Rec.tosptensor().subs
        tempvals = Rec.tosptensor().vals
        #newfilename = file_name[:-4] + '_Recover' + file_name[-4:]
        print("\n" + "The original Tensor is: ")
        print(Ori)
        print("\n" + "The Recovered Tensor is: ")
        print(Rec.data)

    """
    # Reconstruct
    if function_name != '7' and function_name != 'PARAFAC2':
        df = dat1
        for i in range(nv):
            pos = list(map(sum, newsubs == subs[i]))
            idx = pos.index(nd)
            temp = tempvals[idx]
            df.iloc[i, nd] = temp[0]
            # newvals.append(list(tempvals(idx)));
        df.to_csv(newfilename, sep=';', index=0)
    """

    
    return Final_tensor_real

