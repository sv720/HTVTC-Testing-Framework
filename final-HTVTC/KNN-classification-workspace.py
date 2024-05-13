#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from finalAlgoImplementation import final_HTVTC, exploratory_HTVTC_with_intermediate_ground_truth_eval, exploratory_HTVTC_with_intermediate_gt_on_best_val_patches
import regressionmetrics
import copy
import classificationmetrics

quantity = 'EXEC-TIME'

task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data, method = 'cross_validation')
func = crossValidationFunctionGenerator(data_split, algorithm='knn-classification', task=task)
metric = classificationmetrics.indicatorFunction

#Start timer/memory profiler/CPU timer
a = None
start_time = None
if quantity == 'EXEC-TIME':
    import time
    start_time = time.perf_counter_ns()
elif quantity == 'CPU-TIME':
    import time
    start_time = time.process_time_ns()
elif quantity == 'MAX-MEMORY':
    import tracemalloc
    tracemalloc.start()

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

ranges_dict_copy_2 = copy.deepcopy(ranges_dict)
#recommended_combination, history = exploratory_HTVTC_with_intermediate_gt_on_best_val_patches(eval_func=func, ranges_dict=ranges_dict_copy_2, metric=metric, num_best_tc_values_evaluated_at_gt=5, fraction_true_val_to_trigger_patch=0.5)
#recommended_combination, history = exploratory_HTVTC_with_intermediate_ground_truth_eval(eval_func=func, ranges_dict=ranges_dict_copy_2, metric=metric, num_ground_truth_samples= 5, max_completion_cycles=4)
#recommended_combination, history = exploratory_HTVTC_with_intermediate_gt_on_best_val_patches(eval_func=func, ranges_dict=ranges_dict_copy_2, metric=metric, num_best_tc_values_evaluated_at_gt=5, fraction_true_val_to_trigger_patch=10000.0) #THIS IS A PROXI FOR exploratory_HTVTC_with_intermediate_ground_truth_eval


recommended_combination, history = final_HTVTC(eval_func=func, ranges_dict=ranges_dict, metric=metric, max_completion_cycles=5)

#End timer/memory profiler/CPU timer
result = None
if quantity == 'EXEC-TIME':
    end_time = time.perf_counter_ns()
    result = end_time - start_time
elif quantity == 'CPU-TIME':
    end_time = time.process_time_ns()
    result = end_time - start_time
elif quantity == 'MAX-MEMORY':
    _, result = tracemalloc.get_traced_memory()
    tracemalloc.stop()

#Recreate cross-validation generator
data_split = trainTestSplit(binary_data, method = 'cross_validation')
#Find the true loss for the selcted combination
truefunc = crossValidationFunctionGenerator(data_split, algorithm='knn-classification', task=task)   
true_value = truefunc(metric=metric, **recommended_combination)

print(f'hyperparameters: {recommended_combination}')
print(f'history: {history}')
print(f'True value: {true_value}')
print(f'{quantity}: {result}')
if quantity == 'EXEC-TIME':
    print(f'EXEC-TIME in s : {result * (10**(-9))}')
