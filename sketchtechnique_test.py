#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import copy
import classificationmetrics
from sketchtechnique import tensorCompletionSketchingMRP
import numpy as np
from crosstechnique import generateCrossComponents, noisyReconstruction
from generateerrortensor import generateIncompleteErrorTensor


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

reconstructed_tensor_sketch = tensorCompletionSketchingMRP(eval_func=func, ranges_dict=ranges_dict_copy_1, metric=metric,eval_trials=1, known_fraction=0.1, assumed_rank = 1)


ground_truth_tensor, _ = generateIncompleteErrorTensor(eval_func=func, ranges_dict=ranges_dict_copy_2, known_fraction=1, metric=metric, eval_trials=1)

mse_ground_truth_sketch = np.mean((ground_truth_tensor - reconstructed_tensor_sketch)**2)
#mse_sparse_tensor = np.mean((sparse_tensor - reconstructed_tensor)**2)

body, joints, arms = generateCrossComponents(eval_func=func, ranges_dict=ranges_dict_copy_3, metric=metric, eval_trials=1)
completed_tensor_cross = noisyReconstruction(body, joints, arms)

mse_ground_truth_cross = np.mean((ground_truth_tensor - completed_tensor_cross.reshape(11, 2, 1, 11))**2)
ground_truth_squared_magnitude = np.mean((ground_truth_tensor)**2)
print("DEBUG: reconstructed_tensor_sketch = \n ", reconstructed_tensor_sketch)
print("DEBUG: ground_truth_tensor = \n ",ground_truth_tensor)
completed_tensor_cross_squared_magnitude = np.mean((completed_tensor_cross)**2)
reconstructed_tensor_sketch_squared_magnitude = np.mean((reconstructed_tensor_sketch)**2)


print("DEBUG: mse_ground_truth_sketch = ", mse_ground_truth_sketch)
print("DEBUG: mse_ground_truth_cross = ", mse_ground_truth_cross)
print("DEBUG: ground_truth_squared_magnitude = ", ground_truth_squared_magnitude)
print("DEBUG: completed_tensor_cross_squared_magnitude = ", completed_tensor_cross_squared_magnitude)
print("DEBUG: reconstructed_tensor_sketch_squared_magnitude = ", reconstructed_tensor_sketch_squared_magnitude)


