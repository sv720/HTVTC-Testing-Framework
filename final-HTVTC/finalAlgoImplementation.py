#Enable importing code from parent directory
import numpy as np
import copy
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from crosstechnique import generateCrossComponents, noisyReconstruction, noisyReconstruction_modified_experiment2, generateCrossComponents_modified_experiment1, generateCrossComponents_modified_experiment2
from sketchtechnique import tensorCompletionSketchingMRP
from FCTNtechnique import FCTN_TC, FCTN_TC_minmax_feat_scal_norm
from tensorsearch import findBestValues, findBestValues_sort, hyperparametersFromIndices
from generateerrortensor import generateIncompleteErrorTensor
from pyten_completion import pyten_TC
np.random.seed(1)


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
        #print(f'DEBUG: original method: completed tensor = \n {completed_tensor} ')
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        #print(f'DEBUG: original method: bestValue = \n {bestValue} ')

        #print(f'in final_HTVTC bestValue= : {bestValue}')
        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'DEBUG: original method: selected_combination = \n {selected_combination}')
        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        
        
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
        #print(f'DEBUG: replacement w/ ground truth after TC method: completed tensor = \n {completed_tensor} ')
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        #print(f'DEBUG: replacement w/ ground truth after TC method: bestValue = \n {bestValue} ')
        #print(f'in final_HTVTC bestValue= : {bestValue}')
        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'DEBUG: replacement w/ ground truth after TC method: selected_combination = \n {selected_combination}')
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


# exploratory_HTVTC_with_intermediate_ground_truth_eval_on_bestvalues: Third experiment
"""
In experiment 2, we saw that replacing values at random in the completed tensor had little to no effect on the part of the
searchspace explored. 

Hence, instead of evaluating the ground truth at random in the search space, we get a number of top candidates (bestValues)
from the completed tensor and evaluate them using the ground truth. 

The best ground truth amongst these best TC infered candidates is then use to narrow the search space
"""

def exploratory_HTVTC_with_intermediate_ground_truth_eval_on_bestvalues(ranges_dict, eval_func, metric, num_best_tc_values_evaluated_at_gt, **kwargs):

    # Deal with kwargs that are not passed into tensor generation------------------------------------------------------
    kwargskeys = kwargs.keys()
    #The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
    min_interval = 1
    if 'min_interval' in kwargskeys:
        min_interval = kwargs['min_interval']
    #The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
    max_completion_cycles = 4
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
        #print(f' ===== in final_HTVTC cycle : {cycle_num} =====')
        #Perform the tensor completion
        body, joints, arms = generateCrossComponents(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, **kwargs)
        completed_tensor = noisyReconstruction(body, joints, arms)
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        bestValues_TC_Infered = findBestValues_sort(completed_tensor, smallest=True, number_of_values=num_best_tc_values_evaluated_at_gt)

        #print(f'DEBUG: experiment3 method: bestValue = \n {bestValue} ')
        #print(f'DEBUG: experiment3 method: bestValues_TC_Infered = \n {bestValues_TC_Infered} ')

        #print(f'in final_HTVTC bestValue= : {bestValue}')
        #index_list, value_list = bestValue['indices'], bestValue['values']
        index_list_tc_infered, value_lists_tc_infered = bestValues_TC_Infered['indices'], bestValues_TC_Infered['values']

        # for valueCandidate in bestValues_TC_Infered:
        #     current_hyperparameter_values = indexToHyperparameter(random_coords, hyperparameter_values)


        #Obtain hyperparameter from it
        #combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        combinations_tc_infered = hyperparametersFromIndices(index_list_tc_infered, ranges_dict, ignore_length_1=True)
        #print(f'DEBUG: combinations_dbg = \n  {combinations_tc_infered} ')

        evaluation_mode = 'prediction'

        print(f'DEBUG: completed_tensor.size = {completed_tensor.size}')

        true_value_list = []
        for i in range(len(combinations_tc_infered)):
            print(f'DEBUG: i = {i}')
            current_hyperparameter_values = combinations_tc_infered[i]
            true_value_at_coord = eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)
            true_value_list.append(true_value_at_coord)
            print(f'DEBUG: tc_infered_value_at_coord    = {bestValues_TC_Infered["values"][i]} ')
            print(f'DEBUG: true_value_at_coord          = {true_value_at_coord}')


        

        index_of_best_ground_truth_value = true_value_list.index(min(true_value_list)) #TODO: check if this shouldn't be a min (if we are working with a min)...

        selected_combination = combinations_tc_infered[index_of_best_ground_truth_value]
        print("DEBUG: selected_combination = ", selected_combination)
        #print(f'DEBUG: original method: selected_combination = \n {selected_combination}')
        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_lists_tc_infered[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        

        
        #print(f'combinations   : {combinations}') 
        #print(f'combinations[0]: {combinations[0]}') 


        
        # TODO: check if this need changing with new method 
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
            print("DEBUG: below completed tensor is smaller than maximum size of grid-search: doing grid search ")
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
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination':'WARNING RESULT FROM GRID SEARCH - predicted_loss already true', 'method': 'grid search'})
            break
        
        #Only need to update the ranges dict if we are using it in the next loop iteration.
        if cycle_num == max_completion_cycles - 1:
            break
        ranges_dict = update_ranges_dict(ranges_dict, selected_combination, min_interval)
        
    #return the optimal hyperparameter combination as decided by the algorithm-------------------------------------------
    return selected_combination, history


'''
if exploratory_HTVTC_with_intermediate_gt_on_best_val_patches, we perform as we did in exploratory_HTVTC_with_intermediate_ground_truth_eval_on_bestvalues
except that, when we find points amongst our best candidates that have an error that is large (0.5 times the true value at that point), we perform a grid search around that point 
(or these points where we have large error).
The idea here is that if there is a large difference between the true value and the value infered by TC: we need to measure again and search a patch around 
the candidate point

'''
def exploratory_HTVTC_with_intermediate_gt_on_best_val_patches(ranges_dict, eval_func, metric, num_best_tc_values_evaluated_at_gt, fraction_true_val_to_trigger_patch=0.5, **kwargs):

    # Deal with kwargs that are not passed into tensor generation------------------------------------------------------
    kwargskeys = kwargs.keys()
    #The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
    min_interval = 1
    if 'min_interval' in kwargskeys:
        min_interval = kwargs['min_interval']
    #The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
    max_completion_cycles = 4
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
        print(f' ===== in final_HTVTC cycle : {cycle_num} =====')
        #Perform the tensor completion
        body, joints, arms = generateCrossComponents(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, **kwargs)
        completed_tensor = noisyReconstruction(body, joints, arms)
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        bestValues_TC_Infered = findBestValues_sort(completed_tensor, smallest=True, number_of_values=num_best_tc_values_evaluated_at_gt)

        #print(f'DEBUG: experiment3 method: bestValue = \n {bestValue} ')
        #print(f'DEBUG: experiment3 method: bestValues_TC_Infered = \n {bestValues_TC_Infered} ')

        #print(f'in final_HTVTC bestValue= : {bestValue}')
        #index_list, value_list = bestValue['indices'], bestValue['values']
        index_list_tc_infered, value_lists_tc_infered = bestValues_TC_Infered['indices'], bestValues_TC_Infered['values']

        # for valueCandidate in bestValues_TC_Infered:
        #     current_hyperparameter_values = indexToHyperparameter(random_coords, hyperparameter_values)


        #Obtain hyperparameter from it
        #combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        combinations_tc_infered = hyperparametersFromIndices(index_list_tc_infered, ranges_dict, ignore_length_1=True)
        
        #print(f'DEBUG: combinations_dbg = \n  {combinations_tc_infered} ')

        evaluation_mode = 'prediction'

        print(f'DEBUG: completed_tensor.size = {completed_tensor.size}')

        true_value_list = []
        best_value_in_each_patch = []
        combination_of_best_value_in_each_patch = []

        for i in range(len(combinations_tc_infered)):
            print(f'DEBUG: i = {i}')
            current_hyperparameter_values = combinations_tc_infered[i]
            true_value_at_coord = eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)
            true_value_list.append(true_value_at_coord)
            print(f'DEBUG: tc_infered_value_at_coord    = {bestValues_TC_Infered["values"][i]} ')
            print(f'DEBUG: true_value_at_coord          = {true_value_at_coord} ')
            


            if( abs(true_value_at_coord - (bestValues_TC_Infered["values"][i])) > true_value_at_coord*fraction_true_val_to_trigger_patch ): #if we have a large error: perform a grid search around the point
                print("DEBUG: found large error at current_hyperparameter_values = \n", current_hyperparameter_values)
                # print("DEBUG: ranges_dict = \n", ranges_dict)
                #combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
                # print("DEBUG: ranges_dict = \n ", ranges_dict)
                # print("DEBUG: current_hyperparameter_values = \n ", current_hyperparameter_values)
                intervals_dict = {}
                for key in ranges_dict:
                    # print("key = ", key)
                    # print("ranges_dict[key] = ", ranges_dict[key])
                    if 'interval' in ranges_dict[key]: 
                        intervals_dict[key] = ranges_dict[key]['interval'] 
                    # print("--")

                # print("intervals_dict = ", intervals_dict)

               

                patch_ranges_dict = copy.deepcopy(ranges_dict)
                
                print("DEBUG: current_hyperparameter_values \n ", current_hyperparameter_values)
                #print("DEBUG: OLD patch_ranges_dict = \n", patch_ranges_dict)
                for key in patch_ranges_dict:
                    if patch_ranges_dict[key]['type'] == 'INTEGER':
                        if key != 'min_samples_split':
                            patch_ranges_dict[key]['start'] = max(1.0, current_hyperparameter_values[key] - patch_ranges_dict[key]['interval']//2)
                            patch_ranges_dict[key]['end'] = max(1.0, current_hyperparameter_values[key] + patch_ranges_dict[key]['interval']//2)
                            patch_ranges_dict[key]['interval'] = max(1.0, patch_ranges_dict[key]['interval']//2)

                #print("DEBUG: NEW patch_ranges_dict = \n", patch_ranges_dict)

                full_tensor_in_patch , _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=patch_ranges_dict, known_fraction=1, metric=metric, eval_trials=eval_trials, **kwargs)
             
                #print("DEBUG: full_tensor_in_patch = \n ", full_tensor_in_patch)

                #Find best value (true value: not infered)
                bestValue_in_patch = findBestValues(full_tensor_in_patch, smallest=True, number_of_values=1)
                del full_tensor_in_patch
                index_list, value_list = bestValue_in_patch['indices'], bestValue_in_patch['values']
                #Obtain hyperparameter from it
                combinations_in_patch = hyperparametersFromIndices(index_list, patch_ranges_dict, ignore_length_1=True)
                selected_combination_in_patch = combinations_in_patch[0]
                #del combinations_in_patch
                print("DEBUG: bestValue_in_patch = ", bestValue_in_patch)
                print("DEBUG: hyperparametersFromIndices(bestValue_in_patch['indices'], patch_ranges_dict, ignore_length_1=True) = ", hyperparametersFromIndices(bestValue_in_patch['indices'], patch_ranges_dict, ignore_length_1=True))


                best_value_in_each_patch.append(bestValue_in_patch['values'])
                print("DEBUG: selected_combination_in_patch = ", selected_combination_in_patch)
                combination_of_best_value_in_each_patch.append(selected_combination_in_patch)

                # true_value_in_patch = eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)

                print('==================')

        index_of_best_ground_truth_value = true_value_list.index(min(true_value_list)) #TODO: check if this should rly be a max and not a min?!?!

        selected_combination = combinations_tc_infered[index_of_best_ground_truth_value]
        print("DEBUG: BEFORE GETTING VALUE IN PATCH: selected_combination = ", selected_combination)

        #But: if we got a better value in one of the patches: we will use that value instead
        if len(best_value_in_each_patch) > 0:
            if (min(true_value_list) > min(best_value_in_each_patch)):
                print("DEBUG: a value in a patch was better than found in the best TC candidates")
                print("DEBUG: old best value = ", min(true_value_list))
                print("DEBUG: which is with params = ", selected_combination)
                
                index_best_performing_patch = best_value_in_each_patch.index(min(best_value_in_each_patch))
                selected_combination = combination_of_best_value_in_each_patch[index_best_performing_patch]
                print("DEBUG: new best value = ", min(best_value_in_each_patch))
                print("DEBUG: which is with params = ", selected_combination)




        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        
        
 


        
        # TODO: check if this need changing with new method 
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
            print("DEBUG: below completed tensor is smaller than maximum size of grid-search: doing grid search ")
            #print(f'completed_tensor.size =  : {completed_tensor.size}')
            #Generate complete tensor
            full_tensor, _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=1, metric=metric, eval_trials=eval_trials, **kwargs)
            #Find best value (true value: not infered)
            bestValue = findBestValues(full_tensor, smallest=True, number_of_values=1)
            del full_tensor
            index_list, value_list = bestValue['indices'], bestValue['values']
            #Obtain hyperparameter from it
            combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
            selected_combination = combinations[0]
            #print(f'selected_combination (g) = : {selected_combination}')

            #Add to history
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination':'WARNING RESULT FROM GRID SEARCH - predicted_loss already true', 'method': 'grid search'})
            break
        
        #Only need to update the ranges dict if we are using it in the next loop iteration.
        if cycle_num == max_completion_cycles - 1:
            break
        ranges_dict = update_ranges_dict(ranges_dict, selected_combination, min_interval)
        
    #return the optimal hyperparameter combination as decided by the algorithm-------------------------------------------
    return selected_combination, history

"""
final_HTVTC_TSvMRP is the same as final_HTVTC, except that the tensor is no longer being completed with with cross but with Tensor Sketching via Multiple Random Projections tensor completion
"""
def final_HTVTC_TSvMRP(ranges_dict, eval_func, metric, initial_known_fraction, assumed_rank, known_fraction_multiplier = 1, **kwargs):

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
    known_fraction = initial_known_fraction*known_fraction_multiplier
    for cycle_num in range(max_completion_cycles):
        print(f'in final_HTVTC cycle : {cycle_num}')
        #Perform the tensor completion
        completed_tensor = tensorCompletionSketchingMRP(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, known_fraction=known_fraction, assumed_rank=assumed_rank)
        known_fraction = known_fraction_multiplier*np.sum(completed_tensor.shape)/np.product(completed_tensor.shape)#this ensures we sample the same number of points as would in cross (of scaled by multiplier)
        #print(f'DEBUG: original method: completed tensor = \n {completed_tensor} ')
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        #print(f'DEBUG: original method: bestValue = \n {bestValue} ')

        #print(f'in final_HTVTC bestValue= : {bestValue}')
        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'DEBUG: original method: selected_combination = \n {selected_combination}')
        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        
        
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

"""
final_HTVTC_FCTN is the same as final_HTVTC, except that the tensor is no longer being completed with with cross but with Fully-Connected Tensor Network completion
"""
def final_HTVTC_FCTN(ranges_dict, eval_func, metric, initial_known_fraction, assumed_rank_max, known_fraction_multiplier = 1, maxit_fctn = 1000, **kwargs):

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
    known_fraction = initial_known_fraction*known_fraction_multiplier
    for cycle_num in range(max_completion_cycles):
        #print(f'in final_HTVTC cycle : {cycle_num}')
        #Perform the tensor completion

        sparse_tensor, observed_indices = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=known_fraction, metric=metric, eval_trials=eval_trials, **kwargs)
        n_dims = sparse_tensor.ndim
        max_R = np.full((n_dims, n_dims), assumed_rank_max)
        #max_R = np.ones((n_dims,n_dims)) #TODO: think about using assumed_rank_max
        completed_tensor, _ = FCTN_TC(sparse_tensor=sparse_tensor, observed_entries_indices=observed_indices, max_R=max_R, rho=0.1, tol=1e-5,maxit=maxit_fctn)
        known_fraction = known_fraction_multiplier*np.sum(completed_tensor.shape)/np.prod(completed_tensor.shape)#this ensures we sample the same number of points as would in cross (of scaled by multiplier)
        #print(f'DEBUG: original method: completed tensor = \n {completed_tensor} ')
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)

        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'DEBUG: original method: selected_combination = \n {selected_combination}')
        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
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



"""
final_HTVTC_FCTN_minmax_feat_scal_norm is the same as final_HTVTC_FCTN except that FCTN is applied with a min max normalization (of the full sparse tensor) followed by denormalization
"""
def final_HTVTC_FCTN_minmax_feat_scal_norm(ranges_dict, eval_func, metric, initial_known_fraction, assumed_rank_max, known_fraction_multiplier = 1, maxit_fctn = 1000, **kwargs):

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
    known_fraction = initial_known_fraction*known_fraction_multiplier
    for cycle_num in range(max_completion_cycles):
        #print(f'in final_HTVTC cycle : {cycle_num}')
        #Perform the tensor completion

        sparse_tensor, observed_indices = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=known_fraction, metric=metric, eval_trials=eval_trials, **kwargs)
        n_dims = sparse_tensor.ndim
        max_R = np.full((n_dims, n_dims), assumed_rank_max)
        #max_R = np.ones((n_dims,n_dims)) #TODO: think about using assumed_rank_max
        completed_tensor, _ = FCTN_TC_minmax_feat_scal_norm(sparse_tensor=sparse_tensor, observed_entries_indices=observed_indices, max_R=max_R, rho=0.1, tol=1e-5,maxit=maxit_fctn)
        known_fraction = known_fraction_multiplier*np.sum(completed_tensor.shape)/np.prod(completed_tensor.shape)#this ensures we sample the same number of points as would in cross (of scaled by multiplier)
        #print(f'DEBUG: original method: completed tensor = \n {completed_tensor} ')
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)

        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'DEBUG: original method: selected_combination = \n {selected_combination}')
        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
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

"""
final_HTVTC_Tucker_ALS is the same as final_HTVTC except that the tensor is no longer being completed with cross but with Tucker ALS (from tenpy)
"""
def final_HTVTC_Tucker_ALS(ranges_dict, eval_func, metric, initial_known_fraction, assumed_rank=2, known_fraction_multiplier = 1, maxit_fctn = 1000, **kwargs):
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
    known_fraction = initial_known_fraction*known_fraction_multiplier
    for cycle_num in range(max_completion_cycles):
        #print(f'in final_HTVTC cycle : {cycle_num}')
        #Perform the tensor completion

        sparse_tensor, observed_indices = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=known_fraction, metric=metric, eval_trials=eval_trials, **kwargs)
        #n_dims = sparse_tensor.ndim
        sparse_tensor[sparse_tensor == 0] = np.nan
        

        sparse_tensor = np.squeeze(sparse_tensor)
        
        #print("DEBUG: sparse_tensor.shape = ", sparse_tensor.shape)

        #print("DEBUG: sparse_tensor = \n ", sparse_tensor)
        completed_tensor= pyten_TC(sparse_tensor=sparse_tensor, function_name='tucker_als', r=assumed_rank, tol=1e-8,maxiter=maxit_fctn)
        #print("DEBUG: completed_tensor = \n ", completed_tensor)

        known_fraction = known_fraction_multiplier*np.sum(completed_tensor.shape)/np.prod(completed_tensor.shape)#this ensures we sample the same number of points as would in cross (of scaled by multiplier)
        #print(f'DEBUG: original method: completed tensor = \n {completed_tensor} ')
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)

        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'DEBUG: original method: selected_combination = \n {selected_combination}')
        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
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

"""
final_HTVTC_CP_ALS is the same as final_HTVTC except that the tensor is no longer being completed with cross but with CP ALS (from tenpy)
"""
def final_HTVTC_CP_ALS(ranges_dict, eval_func, metric, initial_known_fraction, assumed_rank=2, known_fraction_multiplier = 1, maxit_fctn = 1000, tol=1e-04, **kwargs):
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
    known_fraction = initial_known_fraction*known_fraction_multiplier
    for cycle_num in range(max_completion_cycles):
        #print(f'in final_HTVTC cycle : {cycle_num}')
        #Perform the tensor completion

        sparse_tensor, observed_indices = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=known_fraction, metric=metric, eval_trials=eval_trials, **kwargs)
        #n_dims = sparse_tensor.ndim
        sparse_tensor[sparse_tensor == 0] = np.nan
        

        sparse_tensor = np.squeeze(sparse_tensor)
        
        #print("DEBUG: sparse_tensor.shape = ", sparse_tensor.shape)

        #print("DEBUG: sparse_tensor = \n ", sparse_tensor)
        completed_tensor= pyten_TC(sparse_tensor=sparse_tensor, function_name='cp_als', r=assumed_rank, tol=tol,maxiter=maxit_fctn)
        #print("DEBUG: completed_tensor = \n ", completed_tensor)

        known_fraction = known_fraction_multiplier*np.sum(completed_tensor.shape)/np.prod(completed_tensor.shape)#this ensures we sample the same number of points as would in cross (of scaled by multiplier)
        #print(f'DEBUG: original method: completed tensor = \n {completed_tensor} ')
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)

        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'DEBUG: original method: selected_combination = \n {selected_combination}')
        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
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


"""
final_HTVTC_TNCP is the same as final_HTVTC except that the tensor is no longer being completed with cross but with TNCP (from tenpy)
"""
def final_HTVTC_TNCP(ranges_dict, eval_func, metric, initial_known_fraction, assumed_rank=2, known_fraction_multiplier = 1, maxit_fctn = 500, tol=1e-05, **kwargs):
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
    known_fraction = initial_known_fraction*known_fraction_multiplier
    for cycle_num in range(max_completion_cycles):
        #print(f'in final_HTVTC cycle : {cycle_num}')
        #Perform the tensor completion

        sparse_tensor, observed_indices = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=known_fraction, metric=metric, eval_trials=eval_trials, **kwargs)
        #n_dims = sparse_tensor.ndim
        sparse_tensor[sparse_tensor == 0] = np.nan
        

        sparse_tensor = np.squeeze(sparse_tensor)
        
        #print("DEBUG: sparse_tensor.shape = ", sparse_tensor.shape)

        #print("DEBUG: sparse_tensor = \n ", sparse_tensor)
        completed_tensor= pyten_TC(sparse_tensor=sparse_tensor, function_name='TNCP', r=assumed_rank, tol=tol,maxiter=maxit_fctn)
        #print("DEBUG: completed_tensor = \n ", completed_tensor)
        #print("DEBUG: completed_tensor = \n ", completed_tensor)
        known_fraction = known_fraction_multiplier*np.sum(completed_tensor.shape)/np.prod(completed_tensor.shape)#this ensures we sample the same number of points as would in cross (of scaled by multiplier)
        #print(f'DEBUG: original method: completed tensor = \n {completed_tensor} ')
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)

        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #print(f'DEBUG: original method: selected_combination = \n {selected_combination}')
        true_loss_at_selected_combination = eval_func(metric=metric, **selected_combination)
        #Add to history 
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination': true_loss_at_selected_combination, 'method': 'tensor completion'})
        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
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
#Repeat of the final_HTVTC function that records the timestamps at the end of each cycle
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
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'true_loss_at_selected_combination':'WARNING RESULT FROM GRID SEARCH - predicted_loss already true', 'method': 'grid search'})
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

    
