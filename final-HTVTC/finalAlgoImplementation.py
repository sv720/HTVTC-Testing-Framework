#Enable importing code from parent directory
import numpy as np
import copy
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from crosstechnique import generateCrossComponents, noisyReconstruction, noisyReconstruction_modified_experiment2, generateCrossComponents_modified_experiment1, generateCrossComponents_modified_experiment2
from tensorsearch import findBestValues, hyperparametersFromIndices
from generateerrortensor import generateIncompleteErrorTensor


#Helper function to narrow down the search space through the range dictionary.
def update_ranges_dict(ranges_dict, selected_combination, min_real_interval):
    #Iterate over each hyperparameter
    for hyperparameter in ranges_dict.keys():
        hyp_info = ranges_dict[hyperparameter]
        selected_value = selected_combination[hyperparameter]
        #There is no general way to compare categorical hyperparameters, so these are not interfered with.
        if hyp_info['type'] == 'CATEGORICAL':
            continue
        if hyp_info['type'] in ('INTEGER', 'REAL'):
            #In case a list of values is provided, the list is sorted and a sub-list of half the size, with the selected
            #value at the centre is taken.
            if 'values' in hyp_info.keys():
                no_values = len(hyp_info['values'])
                if no_values == 1:
                    continue
                sorted_values = sorted(hyp_info['values'])
                selected_index = sorted_values.index(selected_value)
                quarter_values = int(round(no_values/4))
                lower_index = max(selected_index - quarter_values, 0)
                #If the upper index is allowed to be equal to the lower index, this would result in an empty list
                upper_index = max( min(selected_index + quarter_values, no_values), lower_index + 1)
               #print(f'DEBUG: lower_index = {lower_index}')
               #print(f'DEBUG: upper_index = {upper_index}')
                ranges_dict[hyperparameter]['values'] = sorted_values[lower_index:upper_index]
            else:
                #Adjust the resolution interval by halving it without letting it fall below the minimum interval size. 
                new_interval = None
                # For an integer hyperparameter, the resolution must be an integer.
                if hyp_info['type'] == 'INTEGER':
                    new_interval = max(hyp_info['interval']//2, 1)
                # For a general real-valued hyperparamater, there is no restriction on the resolution.
                elif hyp_info['type'] == 'REAL':
                    new_interval = max(hyp_info['interval']/2, min_real_interval)
                ranges_dict[hyperparameter]['interval'] = new_interval
                #Adjust endpoints of search space.
                search_space_size = hyp_info['end'] - hyp_info['start']
                # Add and subtract quarter size from the current selected value, so the new limits fit the resolution interval.
                # The new limits should also respect the original limits of the search space.
                quarter_size = search_space_size/4
                new_end = min( round( (selected_value + quarter_size)/new_interval ) * new_interval, hyp_info['end'] )
                new_start = max( round( (selected_value - quarter_size)/new_interval ) * new_interval, hyp_info['start'] )
                ranges_dict[hyperparameter]['end'] = new_end
                ranges_dict[hyperparameter]['start'] = new_start  

               #print(f'DEBUG: new_start = {new_start}')
               #print(f'DEBUG: new_end = {new_end}')          
    #Return updated ranges dict.
    return ranges_dict

                
#The overall implementation.
#List of kwargs:
#   - min_interval: The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
#   - max_completion_cycles: The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
#   - max_size_gridsearch: The maximum number of elements before a grid search can be performed. If 0, this means there will be no grid search.
#   - evaluation_mode: may be needed to configure the evaluation function that populates the tensor.
#   - eval_trials: The number of evaluations of the evaluation function needed to generate one tensor element. Default 1.
#   - tucker_rank_list: The assumed Tucker rank of the completed tensor
def final_HTVTC(ranges_dict, eval_func, metric, **kwargs):

    # Deal with kwargs that are not passed into tensor generation------------------------------------------------------
    kwargskeys = kwargs.keys()
    #The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
    min_interval = 1
    if 'min_interval' in kwargskeys:
        min_interval = kwargs['min_interval']
    #The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
    max_completion_cycles = 1
    if 'max_completion_cycles' in kwargskeys:
        max_completion_cycles = kwargs['max_completion_cycles']
    #The maximum number of elements before a grid search can be performed. If 0, this means there will be no grid search.
    max_size_gridsearch = 0
    if 'max_size_gridsearch' in kwargskeys:
        max_size_gridsearch = kwargs['max_size_gridsearch']
    # The number of evaluations of the evaluation function needed to generate one tensor element.
    eval_trials = 1
    if 'eval_trials' in kwargskeys:
        eval_trials = kwargs['eval_trials']

    #Perform the repeated tensor completions----------------------------------------------------------------------------
    history = []
    selected_combination = None
    for cycle_num in range(max_completion_cycles):
        #print(f'in final_HTVTC cycle : {cycle_num}')
        #Perform the tensor completion
        body, joints, arms = generateCrossComponents(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, **kwargs)
        completed_tensor = noisyReconstruction(body, joints, arms)
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        #print(f'in final_HTVTC bestValue= : {bestValue}')
        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'selected_combination (i) = : {selected_combination}')
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'tensor completion'})
        
        
        #print(f'combinations   : {combinations}') 
        #print(f'combinations[0]: {combinations[0]}') 


        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
            #print("DEBUG: below completed tensor is smaller than maximum size of grid-search: making measurment ")
            #print(f'completed_tensor.size =  : {completed_tensor.size}')
            #Generate complete tensor
            full_tensor, _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=1, metric=metric, eval_trials=eval_trials, **kwargs)
            #Find best value (true value: not infered)
            bestValue = findBestValues(full_tensor, smallest=True, number_of_values=1)
            index_list, value_list = bestValue['indices'], bestValue['values']
            #Obtain hyperparameter from it
            combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
            selected_combination = combinations[0]
            #print(f'selected_combination (g) = : {selected_combination}')

            #Add to history
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'grid search'})
            break
        
        #Only need to update the ranges dict if we are using it in the next loop iteration.
        if cycle_num == max_completion_cycles - 1:
            break
        ranges_dict = update_ranges_dict(ranges_dict, selected_combination, min_interval)
        
    #return the optimal hyperparameter combination as decided by the algorithm-------------------------------------------
    return selected_combination, history

# exploratory_HTVTC_random_coordinates: First experiment
""" attempted to explore random coordinates but work here doesn't make much sense

Picked random coordinates to explore without measuring true values at these points
Even If I did this would pose challenges in terms of convergence as there is not guarantee 
we would converge if we randomly expand the search space
"""
def exploratory_HTVTC_random_coordinates(ranges_dict, ori_ranges_dict, eval_func, metric,  number_random_elements = 1, random_selection_mode='', **kwargs):

    # Deal with kwargs that are not passed into tensor generation------------------------------------------------------
    kwargskeys = kwargs.keys()
    #The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
    min_interval = 1
    if 'min_interval' in kwargskeys:
        min_interval = kwargs['min_interval']
    #The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
    max_completion_cycles = 1
    if 'max_completion_cycles' in kwargskeys:
        max_completion_cycles = kwargs['max_completion_cycles']
    #The maximum number of elements before a grid search can be performed. If 0, this means there will be no grid search.
    max_size_gridsearch = 2
    if 'max_size_gridsearch' in kwargskeys:
        max_size_gridsearch = kwargs['max_size_gridsearch']
    # The number of evaluations of the evaluation function needed to generate one tensor element.
    eval_trials = 1
    if 'eval_trials' in kwargskeys:
        eval_trials = kwargs['eval_trials']

    #Perform the repeated tensor completions----------------------------------------------------------------------------
    history = []
    selected_combination = None
    for cycle_num in range(max_completion_cycles):
        #print(f'======================== in exploratory_HTVTC cycle : {cycle_num} ========================')
        #print(f'DEBUG: ori_ranges_dict = \n {ori_ranges_dict}')
        #Perform the tensor completion
        #print(f'DEBUG: ranges_dict = \n {ranges_dict}')
        body, joints, arms = generateCrossComponents_modified_experiment1(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, ori_ranges_dict=ori_ranges_dict, number_random_elements=number_random_elements, random_selection_mode=random_selection_mode,  **kwargs)       
        
        #DEBUG line to find see if valeus of BJA decomp different without exploration: 
        body_no_random_elems, joints_no_random_elems, arms_no_random_elems = generateCrossComponents_modified_experiment1(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, number_random_elements=0,  **kwargs)
        """
       #print(f'DEBUG: with random points: body  = \n {(body)}')
       #print(f'DEBUG: without random points: body_no_random_elems  = \n {(body_no_random_elems)}')

       #print(f'DEBUG: with random points: joints  = \n {(joints)}')
       #print(f'DEBUG: without random points: joints_no_random_elems  = \n {(joints_no_random_elems)}')

       #print(f'DEBUG: with random points: arms = \n {(arms)}')
       #print(f'DEBUG: without random points: arms_no_random_elems  = \n {(arms_no_random_elems)}')
        """
        #print(f'DEBUG: body = \n {np.array(body)}')
        #print(f'_____')
        #print(f'DEBUG: joints = \n {np.array(joints)}')
        #print(f'_____')
        #print(f'DEBUG: arms = \n {arms}')
        #print(f'================================================================================')
        

        completed_tensor = noisyReconstruction(body, joints, arms)
        completed_tensor_no_random_elems = noisyReconstruction(body_no_random_elems, joints_no_random_elems, arms_no_random_elems)

        
       #print(f'DEBUG: completed_tensor  = \n {(completed_tensor)}')
       #print(f'DEBUG: completed_tensor_no_random_elems  = \n {(completed_tensor_no_random_elems)}')
        #Find best value
        if (np.size(completed_tensor) > 1): 
           #print(f'DEBUG: np.size(completed_tensor) = {np.size(completed_tensor)}')
            bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        else:    
            bestValue = completed_tensor
        if (np.size(completed_tensor) > 1):
            bestValue_no_random_elems = findBestValues(completed_tensor_no_random_elems, smallest=True, number_of_values=1)
        else:    
            bestValue_no_random_elems = completed_tensor_no_random_elems
        #print(f'DEBUG: bestValue = {bestValue}')
        #print(f'DEBUG: bestValue_no_random_elems = {bestValue_no_random_elems}')

        #print(f'in final_HTVTC bestValue= : {bestValue}')
        index_list, value_list = bestValue['indices'], bestValue['values']
        index_list_no_random_elems, value_list_no_random_elems = bestValue_no_random_elems['indices'], bestValue_no_random_elems['values']

        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        combinations_no_random_elems = hyperparametersFromIndices(index_list_no_random_elems, ranges_dict, ignore_length_1=True)

        #print(f'DEBUG: combinations =  \n {combinations}')
       #print(f'DEBUG: combinations_no_random_elems = \n {combinations_no_random_elems}')


        selected_combination = combinations[0]
        selected_combination_no_random_elems = combinations_no_random_elems[0]

        #print(f'DEBUG: selected_combination = \t \t \t {selected_combination}')
        #print(f'DEBUG: selected_combination_no_random_elems = \t {selected_combination_no_random_elems}')



        #print(f'selected_combination (i) = : {selected_combination}')
        #Add to history 
        history.append({'combination': selected_combination_no_random_elems, 'predicted_loss': value_list[0], 'method': 'tensor completion'})
        
        
        #print(f'combinations   : {combinations}') 
        #print(f'combinations[0]: {combinations[0]}') 


        
        #If below limit, perform grid search and break.
        
        """
        if completed_tensor.size < max_size_gridsearch:
            #print("DEBUG: below completed tensor is smaller than maximum size of grid-search: making measurment ")
            #print(f'completed_tensor.size =  : {completed_tensor.size}')
            #Generate complete tensor
            full_tensor, _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=1, metric=metric, eval_trials=eval_trials, **kwargs)
            #Find best value (true value: not infered)
            bestValue_no_random_elems = findBestValues(full_tensor, smallest=True, number_of_values=1)
            index_list, value_list = bestValue_no_random_elems['indices'], bestValue_no_random_elems['values']
            #Obtain hyperparameter from it
            combinations_no_random_elems = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
            selected_combination = combinations[0]
            #print(f'selected_combination (g) = : {selected_combination}')

            #Add to history
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'grid search'})
            break
            
        """
        
        
        #Only need to update the ranges dict if we are using it in the next loop iteration.
        if cycle_num == max_completion_cycles - 1:
            break
        
       #print(f'DEBUG: old ranges_dict= \n {ranges_dict} ')
        ranges_dict = update_ranges_dict(ranges_dict, selected_combination, min_interval)
       #print(f'DEBUG: new ranges_dict= \n {ranges_dict} ')
        #ranges_dict_rand = copy.deepcopy(ranges_dict)
        #ranges_dict_rand = update_ranges_dict(ranges_dict_rand, selected_combination, min_interval)
#
        #print(f'DEBUG: ranges_dict = \n {ranges_dict}')
        #print(f'DEBUG: ranges_dict_rand = \n {ranges_dict_rand}')

        
    #return the optimal hyperparameter combination as decided by the algorithm-------------------------------------------
    return selected_combination, history


# exploratory__HTVTC_with_intermediate_ground_truth_eval: Second experiment
"""
Here use exactly the same method is in final_HTVTC but we evaluate the ground truth at intermediate steps

i.e. instead of narrowing down our search space based entierly based on values evaluated by tensor completion, we evaluate 
ground truth values and insert them into the list of values (instead of the infered ones)

"""
def exploratory_HTVTC_with_intermediate_ground_truth_eval(ranges_dict, eval_func, metric, num_ground_truth_samples, **kwargs):
    
    # Deal with kwargs that are not passed into tensor generation------------------------------------------------------
    kwargskeys = kwargs.keys()
    #The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
    min_interval = 1
    if 'min_interval' in kwargskeys:
        min_interval = kwargs['min_interval']
    #The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
    max_completion_cycles = 1
    if 'max_completion_cycles' in kwargskeys:
        max_completion_cycles = kwargs['max_completion_cycles']
    #The maximum number of elements before a grid search can be performed. If 0, this means there will be no grid search.
    max_size_gridsearch = 0
    if 'max_size_gridsearch' in kwargskeys:
        max_size_gridsearch = kwargs['max_size_gridsearch']
    # The number of evaluations of the evaluation function needed to generate one tensor element.
    eval_trials = 1
    if 'eval_trials' in kwargskeys:
        eval_trials = kwargs['eval_trials']

    #Perform the repeated tensor completions----------------------------------------------------------------------------
    history = []
    selected_combination = None
    for cycle_num in range(max_completion_cycles):
        #print(f'========================= in cycle_num {cycle_num} =========================')
        #print(f'DEBUG: ranges_dict = \n {ranges_dict}')
        #print(f'in final_HTVTC cycle : {cycle_num}')
        #Perform the tensor completion
        
        body, joints, arms = generateCrossComponents(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, **kwargs)
        completed_tensor = noisyReconstruction_modified_experiment2(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, num_ground_truth_samples=num_ground_truth_samples, body=body, joint_matrices=joints, arm_matrices=arms)
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        #print(f'in final_HTVTC bestValue= : {bestValue}')
        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'selected_combination (i) = : {selected_combination}')
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'tensor completion'})
        
        
        #print(f'combinations   : {combinations}') 
        #print(f'combinations[0]: {combinations[0]}') 


        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
            #print("DEBUG: below completed tensor is smaller than maximum size of grid-search: making measurment ")
            #print(f'completed_tensor.size =  : {completed_tensor.size}')
            #Generate complete tensor
            full_tensor, _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=1, metric=metric, eval_trials=eval_trials, **kwargs)
            #Find best value (true value: not infered)
            bestValue = findBestValues(full_tensor, smallest=True, number_of_values=1)
            index_list, value_list = bestValue['indices'], bestValue['values']
            #Obtain hyperparameter from it
            combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
            selected_combination = combinations[0]
            #print(f'selected_combination (g) = : {selected_combination}')

            #Add to history
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'grid search'})
            break
        
        #Only need to update the ranges dict if we are using it in the next loop iteration.
        if cycle_num == max_completion_cycles - 1:
            break
        ranges_dict = update_ranges_dict(ranges_dict, selected_combination, min_interval)
        
    #return the optimal hyperparameter combination as decided by the algorithm-------------------------------------------
    return selected_combination, history
import time

#Repeat of the above function that records the timestamps at the end of each cycle
def final_HTVTC_profiling(ranges_dict, eval_func, metric, **kwargs):

    # Deal with kwargs that are not passed into tensor generation------------------------------------------------------
    kwargskeys = kwargs.keys()
    #The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
    min_interval = 1
    if 'min_interval' in kwargskeys:
        min_interval = kwargs['min_interval']
    #The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
    max_completion_cycles = 1
    if 'max_completion_cycles' in kwargskeys:
        max_completion_cycles = kwargs['max_completion_cycles']
    #The maximum number of elements before a grid search can be performed. If 0, this means there will be no grid search.
    max_size_gridsearch = 0
    if 'max_size_gridsearch' in kwargskeys:
        max_size_gridsearch = kwargs['max_size_gridsearch']
    # The number of evaluations of the evaluation function needed to generate one tensor element.
    eval_trials = 1
    if 'eval_trials' in kwargskeys:
        eval_trials = kwargs['eval_trials']
    minimise = True
    if 'minimise' in kwargskeys:
        minimise = kwargs['minimise']

    #Begin time measurement
    start_time = time.perf_counter_ns()

    #Perform the repeated tensor completions----------------------------------------------------------------------------
    history = []
    selected_combination = None
    #Used for profiling
    timestamps = []

    for cycle_num in range(max_completion_cycles):
        #Perform the tensor completion
        body, joints, arms = generateCrossComponents(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, **kwargs)
        completed_tensor = noisyReconstruction(body, joints, arms)
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=minimise, number_of_values=1)
        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #Add to history
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'tensor completion'})
        #Record timestamp
        timestamp = time.perf_counter_ns()
        time_since_start = timestamp - start_time
        timestamps.append(time_since_start)
        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
            #Generate complete tensor
            full_tensor, _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=1, metric=metric, eval_trials=eval_trials, **kwargs)
            #Find best value
            bestValue = findBestValues(full_tensor, smallest=minimise, number_of_values=1)
            index_list, value_list = bestValue['indices'], bestValue['values']
            #Obtain hyperparameter from it
            combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
            selected_combination = combinations[0]
            #Add to history
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'grid search'})
            #Record timestamp
            timestamp = time.perf_counter_ns()
            time_since_start = timestamp - start_time
            timestamps.append(time_since_start)
            break
        
        #Only need to update the ranges dict if we are using it in the next loop iteration.
        if cycle_num == max_completion_cycles - 1:
            break
        ranges_dict = update_ranges_dict(ranges_dict, selected_combination, min_interval)
        
    #return the optimal hyperparameter combination as decided by the algorithm-------------------------------------------
    return selected_combination, history, timestamps

    
