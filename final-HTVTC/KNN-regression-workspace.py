#Enable importing code from parent directory
import os, sys
from collections import Counter
import statistics
p = os.path.abspath('..')
sys.path.insert(1, p)

import copy

from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from finalAlgoImplementation import final_HTVTC, exploratory_HTVTC_random_coordinates, exploratory_HTVTC_with_intermediate_ground_truth_eval
import regressionmetrics
import classificationmetrics

quantity = 'EXEC-TIME'

task = 'regression'
data = loadData(source='sklearn', identifier='california_housing', task=task)
#data = loadData(source='sklearn', identifier='covertype', task=task)
data_split = trainTestSplit(data, method = 'cross_validation')
func = crossValidationFunctionGenerator(data_split, algorithm='knn-regression', task=task)
metric = regressionmetrics.mae

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
            'values': ['uniform'],
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

number_experiments = 5
for max_completion_cycles in range(2,11):
    for number_random_elements_ground_truth in range(3,12):
        true_values = []
        recommended_combinations = []
        print(f'================================================================================')
        for exp_n in range(1, number_experiments):

            ranges_dict_copy = copy.deepcopy(ranges_dict)
            recommended_combination, history = exploratory_HTVTC_with_intermediate_ground_truth_eval(eval_func=func, ranges_dict=ranges_dict_copy, metric=metric, num_ground_truth_samples= number_random_elements_ground_truth, max_completion_cycles=4)

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
            data_split = trainTestSplit(data, method = 'cross_validation')
            #Find the true loss for the selcted combination
            truefunc = crossValidationFunctionGenerator(data_split, algorithm='knn-regression', task=task)    
            true_value = truefunc(metric=metric, **recommended_combination)

            true_values.append(true_value)
            recommended_combinations.append(recommended_combination)
            # print(f'hyperparameters: {recommended_combination}')
            # print(f'history: {history}')
            # print(f'True value: {true_value}')
            # print(f'{quantity}: {result}')
        print(f'DEBUG: max_completion_cycles =                  {max_completion_cycles}')
        print(f'DEBUG: number_random_elements_ground_truth =    {number_random_elements_ground_truth}')
        print(f'DEBUG: true_values = true_values mean =         {sum(true_values)/len(true_values)} ')
        print(f'DEBUG: true_values = true_values var =          {statistics.variance(true_values)} ')
        dict_counter = Counter(tuple(sorted(d.items())) for d in recommended_combinations)
        most_common_dict = dict_counter.most_common(1)[0][0]
        print(f'DEBUG: modal hyperparams found        =         {most_common_dict}')


"""
ori_ranges_dict_granular = copy.deepcopy(ranges_dict)



#changing the resolution (interval) for integer parameters

for key in ori_ranges_dict_granular:
    if 'interval' in ori_ranges_dict_granular[key]: 
        ori_ranges_dict_granular[key]['interval'] = 1.0


random_selection_mode = 'random_in_original_range' #'from_original_list'

max_completion_cycles = 2
number_random_elements = 5

number_experiments = 5

#print(f'==================== DEBUG: Running exploratory_HTVTC ====================')


for max_completion_cycles in range(2,11):
    for number_random_elements in range(1,6):
        true_values = []
        recommended_combinations = []
        print(f'================================================================================')

        for exp_n in range(1, number_experiments):
            ranges_dict_copy = copy.deepcopy(ranges_dict)
            recommended_combination, history = exploratory_HTVTC(eval_func=func, ranges_dict=ranges_dict_copy, ori_ranges_dict=ori_ranges_dict_granular, number_random_elements=number_random_elements, metric=metric, random_selection_mode=random_selection_mode, max_completion_cycles=max_completion_cycles)
            #recommended_combination, history = final_HTVTC(eval_func=func, ranges_dict=ranges_dict, metric=metric, max_completion_cycles=2)

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
            data_split = trainTestSplit(data, method = 'cross_validation')
            #Find the true loss for the selcted combination
            truefunc = crossValidationFunctionGenerator(data_split, algorithm='knn-regression', task=task)    
            true_value = truefunc(metric=metric, **recommended_combination)

            # print(f'hyperparameters: {recommended_combination}')
            # print(f'history: {history}')
            # print(f'True value: {true_value}')
            # print(f'{quantity}: {result}')

            true_values.append(true_value)
            recommended_combinations.append(recommended_combination)
        print(f'DEBUG: max_completion_cycles =          {max_completion_cycles}')
        print(f'DEBUG: number_random_elements =         {number_random_elements}')
        print(f'DEBUG: true_values = true_values mean = {sum(true_values)/len(true_values)} ')
        print(f'DEBUG: true_values = true_values var =  {statistics.variance(true_values)} ')
        dict_counter = Counter(tuple(sorted(d.items())) for d in recommended_combinations)
        most_common_dict = dict_counter.most_common(1)[0][0]
        print(f'DEBUG: modal hyperparams found        = {most_common_dict}')
    """


            
