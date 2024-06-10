#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)
#sys.path.append('pyten')

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
#import pyten
from pyten_completion import pyten_TC


task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data, method = 'cross_validation')
func = crossValidationFunctionGenerator(data_split, algorithm='knn-classification', task=task)
metric = classificationmetrics.indicatorFunction

ranges_dict = {
        'N': {
            'type': 'INTEGER',
            'start': 1.0,
            'end': 100.0,
            'interval': 10.0,
        },
        'weightingFunction': {
            'type': 'CATEGORICAL',
            'values': ['uniform', 'distance'],
        },
        'distanceFunction': {
            'type': 'CATEGORICAL',
            'values': ['minkowski']
        },
        'p': {
            'type': 'INTEGER',
            'start': 1.0,
            'end': 100.0,
            'interval': 10.0,
        }
    }


ranges_dict_copy_1 = copy.deepcopy(ranges_dict)
ranges_dict_copy_2 = copy.deepcopy(ranges_dict)
ranges_dict_copy_3 = copy.deepcopy(ranges_dict)

#reconstructed_tensor_sketch = tensorCompletionSketchingMRP(eval_func=func, ranges_dict=ranges_dict_copy_1, metric=metric,eval_trials=1, known_fraction=0.1, assumed_rank = 1)
#sparse_tensor, sampled_indices = generateIncompleteErrorTensor(eval_func=func, ranges_dict=ranges_dict, known_fraction=0.1, metric=metric, eval_trials=1, empty_are = 'zero')
#print("DEBUG: sparse_tensor.shape = ", sparse_tensor.shape)
#print("DEBUG: sampled_indices[0] = ", sampled_indices[0])
"""

#TODO: keep this sparse tensor as have the same on in MATLAB
sparse_tensor = np.array([
    [[[0.,0.,0.,0.,0.,0., 0.,0.,0.,0.,0.]],
    [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]],
    [[[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],
    [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]],
    [[[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.09230769, 0.09230769]],
    [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]],
    [[[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],
    [[0.,0.,0.11538462, 0.,0.,0.,0.,0.,0.,0.,0.]]],
    [[[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.13076923, 0.]],
    [[0.,0.,0.,0.,0.,0., 0.,0.,0.,0.13076923, 0.]]],
    [[[0.,0.13076923, 0.,0.,0.,0.,0.13076923, 0.,0.13076923, 0.,0.]],
    [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.13846154, 0.]]],
    [[[0.,0.14615385, 0.,0.,0.,0.14615385, 0.,0.,0.,0., 0.]],
    [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]],
    [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.43076923]],
    [[0., 0., 0., 0., 0., 0., 0.14615385, 0., 0., 0., 0.14615385]]],
    [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.42307692]],
    [[0., 0., 0., 0., 0., 0., 0., 0.14615385, 0., 0., 0.]]],
    [[[0., 0., 0., 0., 0., 0., 0., 0., 0.82307692, 0., 0.]],
    [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]],
    [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    [[0., 0.14615385, 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]
])
"""


ground_truth_tensor = np.array(
    [[[[0.10769231, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385]],
    [[0.10769231, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385]]],
    [[[0.08461538, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769]],
    [[0.08461538, 0.11538462, 0.11538462, 0.12307692, 0.12307692, 0.12307692, 0.12307692, 0.12307692, 0.12307692, 0.12307692, 0.12307692]]],
    [[[0.1, 0.1, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769, 0.09230769]],
    [[0.1, 0.11538462, 0.10769231, 0.10769231, 0.10769231, 0.10769231, 0.10769231, 0.10769231, 0.10769231, 0.10769231, 0.10769231]]], 
    [[[0.1, 0.09230769, 0.08461538, 0.08461538, 0.08461538, 0.08461538, 0.08461538, 0.08461538, 0.08461538, 0.08461538, 0.08461538]],
    [[0.12307692, 0.12307692, 0.11538462, 0.11538462, 0.11538462, 0.11538462, 0.11538462, 0.11538462, 0.11538462, 0.11538462, 0.11538462]]],
    [[[0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923]],
    [[0.12307692, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923]]],
    [[[0.13846154, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923, 0.13076923]],
    [[0.12307692, 0.13846154, 0.13846154, 0.13846154, 0.13846154, 0.13846154, 0.13846154, 0.13846154, 0.13846154, 0.13846154, 0.13846154]]],
    [[[0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385]],
    [[0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385]]],
    [[[0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923]],
    [[0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385]]],
    [[[0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.43076923, 0.42307692]],
    [[0.13846154, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385]]],
    [[[0.82307692, 0.82307692, 0.82307692, 0.82307692, 0.82307692, 0.82307692, 0.82307692, 0.82307692, 0.82307692, 0.82307692, 0.81538462]],
    [[0.13846154, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385]]],
    [[[0.80769231, 0.80769231, 0.80769231, 0.80769231, 0.80769231, 0.80769231, 0.80769231, 0.80769231, 0.80769231, 0.80769231, 0.8,]],
    [[0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385, 0.14615385]]]]
)

#print("DEBUG: ground_truth_tensor.shape = ", ground_truth_tensor.shape)
sampled_fraction = 0.1
assumed_rank = 2
maxiter = 20

print("DEBUG: sampled_fraction = ", sampled_fraction)
print("DEBUG: assumed_rank = ", assumed_rank)
print("DEBUG: maxiter = ", maxiter)
mask = np.random.choice([np.nan, 1], size=ground_truth_tensor.shape, p=[1 - sampled_fraction, sampled_fraction])
#Apply element-wise multiplication to zero out elements based on the mask
sparse_tensor = ground_truth_tensor * mask



#print("DEBUG: sparse_tensor = \n", sparse_tensor)
#print("DEBUG: ground_truth_tensor.shape = ", ground_truth_tensor.shape )
# Tensor dimensions
#dim1, dim2, dim3, dim4 = 11, 1, 1, 11

# Generate a list of all observed indices
#sampled_indices = [(i, j, k, l) for i in range(dim1) for j in range(dim2) for k in range(dim3) for l in range(dim4)]
#sampled_indices = [(0, 0, 0, 0), (8, 0, 0, 5), (1, 0, 0, 6), (4, 0, 0, 6), (8, 0, 0, 6), (7, 0, 0, 7), (10, 0, 0, 8), (3, 0, 0, 9), (5, 0, 0, 9), (7, 0, 0, 9), (2, 0, 0, 10), (9, 0, 0, 10)]
sampled_indices = np.argwhere(sparse_tensor != 0.0) 
sampled_indices = [tuple(idx) for idx in sampled_indices]
#print("DEBUG: sampled_indices = ", sampled_indices)




n_dims = sparse_tensor.ndim
#max_R = np.ones((n_dims,n_dims))
#max_R = np.full((n_dims, n_dims), assumed_rank)
#print("DEBUG: n_dims = ", n_dims)
#print("DEBUG: sparse_tensor.shape = ", sparse_tensor.shape)
#print("DEBUG: max_R = ", max_R)

sparse_tensor = sparse_tensor.reshape(11,2,11)
function_name = 'tucker_als'
#r = 2
tol = 1e-8
init = 'random'
#subs = list(np.ndindex(sparse_tensor.shape))
#vals = np.array([sparse_tensor[idx] for idx in subs])
#vals = vals.reshape(242,1)
#omega = None
#recover = None
#printitn = 0
#file_name = None

start_time = time.perf_counter_ns()
reconsturucted_tensor_tucker_als = pyten_TC(sparse_tensor=sparse_tensor, function_name='tucker_als', r=assumed_rank, tol=tol, maxiter=maxiter)
#reconstructed_tensor_FCTN, _ = FCTN_TC(sparse_tensor, sampled_indices, max_R = max_R, maxit=1000)
end_time = time.perf_counter_ns()

exec_time = end_time - start_time
ground_truth_tensor, _ = generateIncompleteErrorTensor(eval_func=func, ranges_dict=ranges_dict_copy_2, known_fraction=1, metric=metric, eval_trials=1)

#mse_ground_truth_sketch = np.mean((ground_truth_tensor - reconstructed_tensor_sketch)**2)
#mse_sparse_tensor = np.mean((sparse_tensor - reconstructed_tensor)**2)

#body, joints, arms = generateCrossComponents(eval_func=func, ranges_dict=ranges_dict_copy_3, metric=metric, eval_trials=1)
#completed_tensor_cross = noisyReconstruction(body, joints, arms)

#mse_ground_truth_cross = np.mean((ground_truth_tensor - completed_tensor_cross.reshape(11, 2, 1, 11))**2)
#mse_ground_truth_FCTN = np.mean((ground_truth_tensor - reconstructed_tensor_FCTN)**2)
mse_ground_truth_tucker_als = np.mean((ground_truth_tensor.reshape(11,2,11)-reconsturucted_tensor_tucker_als)**2)

ground_truth_squared_magnitude = np.mean((ground_truth_tensor)**2)
#print("DEBUG: reconstructed_tensor_sketch.shape = \n ", reconstructed_tensor_sketch.shape)
#print("DEBUG: ground_truth_tensor         = \n ",ground_truth_tensor)
#print("DEBUG: completed_tensor_cross.shape      = \n", completed_tensor_cross.shape)
#print("DEBUG: reconstructed_tensor_FCTN   = \n ", reconstructed_tensor_FCTN)

#completed_tensor_cross_squared_magnitude = np.mean((completed_tensor_cross)**2)
#reconstructed_tensor_sketch_squared_magnitude = np.mean((reconstructed_tensor_sketch)**2)
#reconstructed_tensor_FCTN_squared_magnitude = np.mean((reconstructed_tensor_FCTN)**2)
reconsturucted_tensor_tucker_als_squared_magnitude = np.mean((reconsturucted_tensor_tucker_als)**2)




#print("DEBUG: mse_ground_truth_sketch   = ", mse_ground_truth_sketch)
#print("DEBUG: mse_ground_truth_cross    = ", mse_ground_truth_cross)

print("DEBUG: ground_truth_squared_magnitude = ", ground_truth_squared_magnitude)
#print("DEBUG: completed_tensor_cross_squared_magnitude      = ", completed_tensor_cross_squared_magnitude)

#print("DEBUG: reconstructed_tensor_sketch_squared_magnitude = ", reconstructed_tensor_sketch_squared_magnitude)
#print("DEBUG: reconstructed_tensor_FCTN_squared_magnitude   = ", reconstructed_tensor_FCTN_squared_magnitude)
#print("DEBUG: mse_ground_truth_FCTN     = ", mse_ground_truth_FCTN)
print("DEBUG: mse_ground_truth_tucker_als = ", mse_ground_truth_tucker_als)
print("Power ratio (Rec/GT) = ", reconsturucted_tensor_tucker_als_squared_magnitude/ground_truth_squared_magnitude)
print("DEBUG: EXEC_TIME (ms) = ", exec_time * (10**(-6)))


#print("DEBUG: ground_truth_tensor = \n", ground_truth_tensor)
#print("DEBUG: reconstructed_tensor_FCTN (from 10 percent samples) = \n", reconstructed_tensor_FCTN)

