from tensorsearch import findBestValues
import tensorly as tl
import numpy as np
import random
import itertools

#random_selection_mode = 'random_in_original_range' #'from_original_list'


#SUPPORT FUNCTION====Convert tensor index to corresponding hyperparameter values based on range dict=============
def indexToHyperparameter(index, value_lists):
    hyperparameter_values = {}
    # Obtain the tensor index and the hyperparameter values to pass to eval_func
    dimension_index = 0
    for key in value_lists.keys():
        value_list = value_lists[key]
        if len(value_list) == 1:
            hyperparameter_values[key] = value_list[0]
        else:
            value_index = int(index[dimension_index])
            hyperparameter_values[key] = value_list[value_index]
            dimension_index += 1
    return hyperparameter_values

#MAIN FUNCTION====Uses the cross technique (Zhang, 2019) to generate the subtensors (body, arm, joint)========================
def generateCrossComponents_modified_experiment1(eval_func, ranges_dict, metric, ori_ranges_dict = {}, number_random_elements=0, random_selection_mode = '',  **kwargs):

    # Obtain the evaluation mode for the machine learning model (prediction, probability or raw score)----------------------------------
    evaluation_mode = 'prediction'
    if 'evaluation_mode' in kwargs.keys():
        evaluation_mode = kwargs['evaluation_mode']
    # Obtain the number of evaluation function trials (default 1)------------------------------------------------------------------------
    eval_trials = 1
    if 'eval_trials' in kwargs.keys():
        eval_trials = kwargs['eval_trials']

    # Obtain the full lists of values for each hyperparameter from the provided ranges/provided list of values--------------------------
    hyperparameter_values = {}
    tensor_dimensions_list = []
    tensor_elements = 1

    if (number_random_elements > 0):
        for key in ori_ranges_dict.keys():
            ori_info = ori_ranges_dict[key]
            ori_value_list = None
            if 'values' in ori_info.keys():
                ori_value_list = ori_info['values']
            else:
                ori_start = float(ori_info['start'])
                ori_end = float(ori_info['end'])
                ori_interval = float(ori_info['interval'])
                #print(f'DEBUG: ori_info = \n {ori_info}')

                ori_value_list = np.linspace(ori_start, ori_end, int(round((ori_end-ori_start)/ori_interval, 0))+1)

    for key in ranges_dict.keys():
        info = ranges_dict[key]
        value_list = None
        #If the values are already provided as a list
        if 'values' in info.keys():
            value_list = info['values']
        #If the values are provided as a range
        else:
            start = float(info['start'])
            end = float(info['end'])
            interval = float(info['interval'])
            value_list = np.linspace(start, end, int(round((end-start)/interval, 0))+1)
        

            if (number_random_elements > 0) and (len(value_list) >= number_random_elements): #here we select random values to explore
                #to not increase the search space - remove elements at random from value_list

                ori_info = ori_ranges_dict[key]
                ori_start = float(ori_info['start'])
                ori_end = float(ori_info['end'])

                #print(f'DEBUG: at first: value_list = \n {value_list}')
                #print(f'DEBUG: ori_value_list = \n {ori_value_list}')
                if (len(value_list) != len(ori_value_list)) or (value_list != ori_value_list).all():

                    #print(f'DEBUG: ori_start = {ori_start}')
                    #print(f'DEBUG: ori_end = {ori_end}')

                    for _ in range(number_random_elements):
                        random_ablation_index = np.random.randint(0, len(value_list))
                        #random_ablation_elem = random.choice(value_list)
                        #print(f'DEBUG: random_ablation_elem = {value_list[random_ablation_index]}')
                        value_list = np.delete(value_list, random_ablation_index)
                        #print(f'DEBUG: after ablation: value_list = \n {value_list}')

                    #select points from original (large) search space to explore     
                    
                    if random_selection_mode == 'from_original_list':
                        rand_selected_ori_values = np.random.choice(ori_value_list, size=number_random_elements, replace=False) #THIS IS TO KEEP ELEMENTS
                    elif random_selection_mode =='random_in_original_range':
                        rand_selected_ori_values = [random.uniform(ori_start, ori_end) for _ in range(number_random_elements)]
                    else: 
                        raise ValueError("random_selection_mode not set to a valid value")
                        

            
                    #print(f'DEBUG: random values added = \n {rand_selected_ori_values}')

                    

                    value_list = np.concatenate((value_list, rand_selected_ori_values))
                    value_list.sort()
                    #print(f'DEBUG: newly generated value_list = \n {value_list}')
                    #print("====================================") 

        # Update outer values
        hyperparameter_values[key] = value_list
        
        if len(value_list) == 1:
            continue

            
        tensor_dimensions_list.append(len(value_list))
        tensor_elements *= len(value_list)
    #print(f'DEBUG: hyperparameter_values = \n {hyperparameter_values}')

    #tensor_elements = {**tensor_elements, **ori_ten}

     

    # Obtain the Tucker rank: if no rank is provided, the default is 1 for each dimension-------------------------------------------------
    tucker_rank_list = [1]*len(tensor_dimensions_list)
    if 'tucker_rank_list' in kwargs.keys():
        tucker_rank_list = kwargs['tucker_rank_list']
        
        if len(tucker_rank_list) != len(tensor_dimensions_list):
            raise(ValueError('The rank list length must be equal to the number of dimensions.'))
        
    tensor_dimensions_tuple = tuple(tensor_dimensions_list)

    # Generate body----------------------------------------------------------------------------------------------------------------------
    # Generate tensor indices of all elements within the body region
    value_lists = []
    for i in range(len(tucker_rank_list)):
        value_lists.append([el for el in range(tucker_rank_list[i])])
    body_indices = list(itertools.product(*value_lists))

    # Assign to each position in the body region using result of eval_func averaged over multiple trials
    body = np.zeros(tuple(tucker_rank_list))
    #print(f'DEBUG: hyperparameter_values = \n {hyperparameter_values}')
    #print(f'DEBUG: body_indices = \n {body_indices}')
    #print(f'================== DEBUG: number_random_elements = {number_random_elements} ==================') 
    for tensor_index in body_indices:
        current_hyperparameter_values = indexToHyperparameter(tensor_index, hyperparameter_values)
        #print(f'DEBUG: tensor_index = \n {tensor_index}')
        #print(f'DEBUG: hyperparameter_values = \n {hyperparameter_values}')

        #print(f'DEBUG: current_hyperparameter_values = {current_hyperparameter_values} \n')
        #print(f'DEBUG: tensor_index = {tensor_index} ')
        #print(f'------- \n')
        eval_result_avg = 0
        for trial in range(eval_trials):
            #print('_____________________________')
            #print(f'current_hyperparameter_values = \n {current_hyperparameter_values}')
            #print(f'metric = \n {metric}')
            #print(f'evaluation_mode = \n {evaluation_mode}')
            #print(f'eval_trials = \n {eval_trials}')
            eval_result_avg += eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
        body[tuple(tensor_index)] = eval_result_avg

    # Generate arms and joints------------------------------------------------------------------------------------------------------------
    arms = []
    joints = []
    # Find the arms and joints along each dimension
    for dimension_index in range(len(tensor_dimensions_tuple)):
        #print(f'DEBUG: number_random_elements = {number_random_elements} dimension_index = {dimension_index} ')
        dimension_rank = tucker_rank_list[dimension_index]
        dimension_size = tensor_dimensions_tuple[dimension_index]
        truncated_rank_list = tucker_rank_list[:dimension_index] + tucker_rank_list[dimension_index+1:]
        #Generate indices of all possible fibres along the dimension that could generate the arms and hinges
        value_lists = []
        for i in range(len(truncated_rank_list)):
            value_lists.append([el for el in range(truncated_rank_list[i])])
        joint_base_indices = list(itertools.product(*value_lists))
        #Randomly sample from the fibre indices to choose the arms and hinges
        no_samples = min(len(joint_base_indices), dimension_rank)
        sampled_indices = random.sample(joint_base_indices, no_samples)
        sorted_samples = sorted(sampled_indices)
        # Generate joint and arm matrices based on the sampled fibres
        joint_matrix = np.zeros((dimension_rank, no_samples))
        arm_matrix = np.zeros((dimension_size, no_samples))
        #col tracks the index of the column of the arm/joint matrix that is being written to
        col = 0
        for base_index in sorted_samples:
            # Generate joint and part of arm from body values
            for seq in range(dimension_rank):
                new_index = base_index[:dimension_index] + (seq,) + base_index[dimension_index:]
                joint_matrix[seq, col] = body[new_index]
                arm_matrix[seq, col] = body[new_index]
            # Generate rest of the arm by evaluating the values
            for seq in range(dimension_rank, dimension_size):
                new_index = base_index[:dimension_index] + (seq,) + base_index[dimension_index:]
                current_hyperparameter_values = indexToHyperparameter(new_index, hyperparameter_values)
                # Assign to matrix using result of eval_func averaged over multiple trials
                eval_result_avg = 0
                for trial in range(eval_trials):
                    eval_result_avg += eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
                arm_matrix[seq, col] = eval_result_avg
            col += 1
        
        #print(f'---')
        #print(f'DEBUG: arm matrix = \n {arm_matrix}')
        #print(f'---')

        arms.append(arm_matrix)
        joints.append(joint_matrix)
            
    return body, joints, arms


#MAIN FUNCTION====Uses the cross technique (Zhang, 2019) to generate the subtensors (body, arm, joint)========================
def generateCrossComponents_modified_experiment2(eval_func, ranges_dict, metric, **kwargs):

    # print(f'DEBUG: eval_func = \n {eval_func}')
    # print(f'DEBUG: ranges_dict = \n {ranges_dict}')
    # Obtain the evaluation mode for the machine learning model (prediction, probability or raw score)----------------------------------
    evaluation_mode = 'prediction'
    if 'evaluation_mode' in kwargs.keys():
        evaluation_mode = kwargs['evaluation_mode']
    # Obtain the number of evaluation function trials (default 1)------------------------------------------------------------------------
    eval_trials = 1
    if 'eval_trials' in kwargs.keys():
        eval_trials = kwargs['eval_trials']

    # Obtain the full lists of values for each hyperparameter from the provided ranges/provided list of values--------------------------
    hyperparameter_values = {}
    tensor_dimensions_list = []
    tensor_elements = 1

    for key in ranges_dict.keys():
        info = ranges_dict[key]
        value_list = None
        #If the values are already provided as a list
        if 'values' in info.keys():
            value_list = info['values']
        #If the values are provided as a range
        else:
            start = float(info['start'])
            end = float(info['end'])
            interval = float(info['interval'])
            value_list = np.linspace(start, end, int(round((end-start)/interval, 0))+1)
        # Update outer values
        hyperparameter_values[key] = value_list
        if len(value_list) == 1:
            continue
        tensor_dimensions_list.append(len(value_list))
        tensor_elements *= len(value_list)

    # Obtain the Tucker rank: if no rank is provided, the default is 1 for each dimension-------------------------------------------------
    tucker_rank_list = [1]*len(tensor_dimensions_list)
    if 'tucker_rank_list' in kwargs.keys():
        tucker_rank_list = kwargs['tucker_rank_list']
        
        if len(tucker_rank_list) != len(tensor_dimensions_list):
            raise(ValueError('The rank list length must be equal to the number of dimensions.'))
        
    tensor_dimensions_tuple = tuple(tensor_dimensions_list)

    # Generate body----------------------------------------------------------------------------------------------------------------------
    # Generate tensor indices of all elements within the body region
    value_lists = []
    for i in range(len(tucker_rank_list)):
        value_lists.append([el for el in range(tucker_rank_list[i])])
    body_indices = list(itertools.product(*value_lists))

    # Assign to each position in the body region using result of eval_func averaged over multiple trials
    body = np.zeros(tuple(tucker_rank_list))
    for tensor_index in body_indices:
        current_hyperparameter_values = indexToHyperparameter(tensor_index, hyperparameter_values)
        eval_result_avg = 0
        for trial in range(eval_trials):
             #print(f'DEBUG: evaluating ground truth 1 ---------------------------------------------------')
             #print(f'DEBUG: current_hyperparameter_values = \n {current_hyperparameter_values}')
            eval_result_tmp = eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
             #print(f'DEBUG: result = \n {eval_result_tmp}')
            eval_result_avg += eval_result_tmp
        body[tuple(tensor_index)] = eval_result_avg
        # print(f'DEBUG: body = \n {body}')
     #print(f'DEBUG: ---------------------------------------------------')
    # Generate arms and joints------------------------------------------------------------------------------------------------------------
    arms = []
    joints = []
    # Find the arms and joints along each dimension
    for dimension_index in range(len(tensor_dimensions_tuple)):
        dimension_rank = tucker_rank_list[dimension_index]
        dimension_size = tensor_dimensions_tuple[dimension_index]
        truncated_rank_list = tucker_rank_list[:dimension_index] + tucker_rank_list[dimension_index+1:]
        #Generate indices of all possible fibres along the dimension that could generate the arms and hinges
        value_lists = []
        for i in range(len(truncated_rank_list)):
            value_lists.append([el for el in range(truncated_rank_list[i])])
        joint_base_indices = list(itertools.product(*value_lists))
        #Randomly sample from the fibre indices to choose the arms and hinges
        no_samples = min(len(joint_base_indices), dimension_rank)
        sampled_indices = random.sample(joint_base_indices, no_samples)
        sorted_samples = sorted(sampled_indices)
        # Generate joint and arm matrices based on the sampled fibres
        joint_matrix = np.zeros((dimension_rank, no_samples))
        arm_matrix = np.zeros((dimension_size, no_samples))
        #col tracks the index of the column of the arm/joint matrix that is being written to
        col = 0
        for base_index in sorted_samples:
            # Generate joint and part of arm from body values
            for seq in range(dimension_rank):
                new_index = base_index[:dimension_index] + (seq,) + base_index[dimension_index:]
                joint_matrix[seq, col] = body[new_index]
                arm_matrix[seq, col] = body[new_index]
            # Generate rest of the arm by evaluating the values
            for seq in range(dimension_rank, dimension_size):
                new_index = base_index[:dimension_index] + (seq,) + base_index[dimension_index:]
                current_hyperparameter_values = indexToHyperparameter(new_index, hyperparameter_values)
                # Assign to matrix using result of eval_func averaged over multiple trials
                eval_result_avg = 0
                for trial in range(eval_trials):
                     #print(f'DEBUG: evaluating ground truth')
                     #print(f'DEBUG: current_hyperparameter_values = \n {current_hyperparameter_values}')
                    eval_result_tmp = eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
                    eval_result_avg += eval_result_tmp
                     #print(f'DEBUG: result = \n {eval_result_tmp}')

                arm_matrix[seq, col] = eval_result_avg
            col += 1

        arms.append(arm_matrix)
        joints.append(joint_matrix)
            
    return body, joints, arms


#MAIN FUNCTION====Uses the cross technique (Zhang, 2019) to generate the subtensors (body, arm, joint)========================
def generateCrossComponents(eval_func, ranges_dict, metric, **kwargs):

    #sample_counter = 0 #TODO: delete this (for debug)
    # Obtain the evaluation mode for the machine learning model (prediction, probability or raw score)----------------------------------
    evaluation_mode = 'prediction'
    if 'evaluation_mode' in kwargs.keys():
        evaluation_mode = kwargs['evaluation_mode']
    # Obtain the number of evaluation function trials (default 1)------------------------------------------------------------------------
    eval_trials = 1
    if 'eval_trials' in kwargs.keys():
        eval_trials = kwargs['eval_trials']

    # Obtain the full lists of values for each hyperparameter from the provided ranges/provided list of values--------------------------
    hyperparameter_values = {}
    tensor_dimensions_list = []
    tensor_elements = 1

    for key in ranges_dict.keys():
        info = ranges_dict[key]
        value_list = None
        #If the values are already provided as a list
        if 'values' in info.keys():
            value_list = info['values']
        #If the values are provided as a range
        else:
            start = float(info['start'])
            end = float(info['end'])
            interval = float(info['interval'])
            value_list = np.linspace(start, end, int(round((end-start)/interval, 0))+1)
        # Update outer values
        hyperparameter_values[key] = value_list
        if len(value_list) == 1:
            continue
        tensor_dimensions_list.append(len(value_list))
        tensor_elements *= len(value_list)

    # Obtain the Tucker rank: if no rank is provided, the default is 1 for each dimension-------------------------------------------------
    tucker_rank_list = [1]*len(tensor_dimensions_list)
    if 'tucker_rank_list' in kwargs.keys():
        tucker_rank_list = kwargs['tucker_rank_list']
        
        if len(tucker_rank_list) != len(tensor_dimensions_list):
            raise(ValueError('The rank list length must be equal to the number of dimensions.'))
        
    tensor_dimensions_tuple = tuple(tensor_dimensions_list)

    # Generate body----------------------------------------------------------------------------------------------------------------------
    # Generate tensor indices of all elements within the body region
    value_lists = []
    for i in range(len(tucker_rank_list)):
        value_lists.append([el for el in range(tucker_rank_list[i])])
    body_indices = list(itertools.product(*value_lists))

    # Assign to each position in the body region using result of eval_func averaged over multiple trials
    body = np.zeros(tuple(tucker_rank_list))
    for tensor_index in body_indices:
        current_hyperparameter_values = indexToHyperparameter(tensor_index, hyperparameter_values)
        eval_result_avg = 0
        for trial in range(eval_trials):
            #print(f'DEBUG: current_hyperparameter_values = \n {current_hyperparameter_values}')
            eval_result_avg += eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
            #sample_counter += 1
        body[tuple(tensor_index)] = eval_result_avg

    # Generate arms and joints------------------------------------------------------------------------------------------------------------
    arms = []
    joints = []
    # Find the arms and joints along each dimension
    for dimension_index in range(len(tensor_dimensions_tuple)):
        dimension_rank = tucker_rank_list[dimension_index]
        dimension_size = tensor_dimensions_tuple[dimension_index]
        truncated_rank_list = tucker_rank_list[:dimension_index] + tucker_rank_list[dimension_index+1:]
        #Generate indices of all possible fibres along the dimension that could generate the arms and hinges
        value_lists = []
        for i in range(len(truncated_rank_list)):
            value_lists.append([el for el in range(truncated_rank_list[i])])
        joint_base_indices = list(itertools.product(*value_lists))
        #Randomly sample from the fibre indices to choose the arms and hinges
        no_samples = min(len(joint_base_indices), dimension_rank)
        sampled_indices = random.sample(joint_base_indices, no_samples)
        sorted_samples = sorted(sampled_indices)
        # Generate joint and arm matrices based on the sampled fibres
        joint_matrix = np.zeros((dimension_rank, no_samples))
        arm_matrix = np.zeros((dimension_size, no_samples))
        #col tracks the index of the column of the arm/joint matrix that is being written to
        col = 0
        for base_index in sorted_samples:
            # Generate joint and part of arm from body values
            for seq in range(dimension_rank):
                new_index = base_index[:dimension_index] + (seq,) + base_index[dimension_index:]
                joint_matrix[seq, col] = body[new_index]
                arm_matrix[seq, col] = body[new_index]
            # Generate rest of the arm by evaluating the values
            for seq in range(dimension_rank, dimension_size):
                new_index = base_index[:dimension_index] + (seq,) + base_index[dimension_index:]
                current_hyperparameter_values = indexToHyperparameter(new_index, hyperparameter_values)
                # Assign to matrix using result of eval_func averaged over multiple trials
                eval_result_avg = 0
                for trial in range(eval_trials):
                    eval_result_avg += eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
                    #sample_counter += 1
                arm_matrix[seq, col] = eval_result_avg
            col += 1

        arms.append(arm_matrix)
        joints.append(joint_matrix)

    #print("DEBUG: sample_counter = ", sample_counter)
    return body, joints, arms

#MAIN FUNCTION====Noiseless reconstruction of the original tensor according to Zhang, 2019=============
def noiselessReconstruction(body, joint_matrices, arm_matrices):
    R_matrices = []
    for index in range(len(joint_matrices)):
        Y_arm_i = arm_matrices[index]
        Y_joint_i = joint_matrices[index]
        R_i = np.matmul(Y_arm_i, np.linalg.pinv(Y_joint_i))
        R_matrices.append(R_i)
    return tl.tucker_tensor.tucker_to_tensor((body, R_matrices))


#MAIN FUNCTION====Reconstruction accounting for noise according to Zhang, 2019==========================
def noisyReconstruction(body, joint_matrices, arm_matrices):
    R_matrices = []
    #Iterate over tensor dimensions
    for index in range(len(joint_matrices)):
        #Prepare matrices
        body_unfolding = tl.unfold(body, mode=index)
        joint_matrix = joint_matrices[index]
        arm_matrix = arm_matrices[index]
        #Calculate SVDs
        U, _, _ = np.linalg.svd(body_unfolding, full_matrices=True)
        _, _, Vh = np.linalg.svd(arm_matrix, full_matrices=True)
        #V contains the right singular vectors as columns
        V = Vh.T
        #Perform rotations
        A = np.matmul(arm_matrix, V)
        J = np.matmul(np.matmul(U.T, joint_matrix), V)
        #Identify value of r by varying s
        max_s = min(np.shape(J))
        r = 0
        product = None
        for s in range(max_s, 0, -1):
            #Condition 1
            trunc_J = J[:s, :s]
            #If all elements are zero, we cannot invert so set diagonal elements
            #to a small value to avoid an error
            if not np.any(trunc_J):
                for diag_ind in range(s):
                    trunc_J[diag_ind, diag_ind] = 1e-10
            if np.linalg.matrix_rank(trunc_J) < s:
                continue
            #Condition 2
            trunc_A = A[:, :s]
            # Lambda calculated using recommended c=3
            lambda_val = 3*(A.shape[0])/(J.shape[0])
            J_trunc_inv = np.linalg.inv(trunc_J)
            product = np.matmul(trunc_A, J_trunc_inv)
            #Calculate matrix spectral norm via svd (largest singular value)
            _, svals, _ = np.linalg.svd(product)
            spectral_norm = max(svals)
            if spectral_norm > lambda_val:
                continue
            r = s
        #Calculate R
        trunc_V = V[:r, :]
        R = np.matmul(product, trunc_V)
        R_matrices.append(R)
    return tl.tucker_tensor.tucker_to_tensor((body, R_matrices))


def noisyReconstruction_modified_experiment2(eval_func, ranges_dict, metric, num_ground_truth_samples, body, joint_matrices, arm_matrices):
    R_matrices = []
    #Iterate over tensor dimensions
    for index in range(len(joint_matrices)):
        #Prepare matrices
        body_unfolding = tl.unfold(body, mode=index)
        joint_matrix = joint_matrices[index]
        arm_matrix = arm_matrices[index]
        #Calculate SVDs
        U, _, _ = np.linalg.svd(body_unfolding, full_matrices=True)
        _, _, Vh = np.linalg.svd(arm_matrix, full_matrices=True)
        #V contains the right singular vectors as columns
        V = Vh.T
        #Perform rotations
        A = np.matmul(arm_matrix, V)
        J = np.matmul(np.matmul(U.T, joint_matrix), V)
        #Identify value of r by varying s
        max_s = min(np.shape(J))
        r = 0
        product = None
        for s in range(max_s, 0, -1):
            #Condition 1
            trunc_J = J[:s, :s]
            #If all elements are zero, we cannot invert so set diagonal elements
            #to a small value to avoid an error
            if not np.any(trunc_J):
                for diag_ind in range(s):
                    trunc_J[diag_ind, diag_ind] = 1e-10
            if np.linalg.matrix_rank(trunc_J) < s:
                continue
            #Condition 2
            trunc_A = A[:, :s]
            # Lambda calculated using recommended c=3
            lambda_val = 3*(A.shape[0])/(J.shape[0])
            J_trunc_inv = np.linalg.inv(trunc_J)
            product = np.matmul(trunc_A, J_trunc_inv)
            #Calculate matrix spectral norm via svd (largest singular value)
            _, svals, _ = np.linalg.svd(product)
            spectral_norm = max(svals)
            if spectral_norm > lambda_val:
                continue
            r = s
        #Calculate R
        trunc_V = V[:r, :]
        R = np.matmul(product, trunc_V)
        R_matrices.append(R)

    final_tensor = tl.tucker_tensor.tucker_to_tensor((body, R_matrices))
    # print(f'DEBUG: final_tensor = \n {final_tensor}')

    # ======================= CODE TO MAKE GROUND TRUTH OBSERVATION AND INSERTING THEM INTO THE COMPLETED TENSOR

    # Obtain the full lists of values for each hyperparameter from the provided ranges/provided list of values--------------------------
    hyperparameter_values = {}
    tensor_parameters_lists = []
    tensor_elements = 1

    for key in ranges_dict.keys():
        info = ranges_dict[key]
        value_list = None
        #If the values are already provided as a list
        if 'values' in info.keys():
            value_list = info['values']
        #If the values are provided as a range
        else:
            start = float(info['start'])
            end = float(info['end'])
            interval = float(info['interval'])
            value_list = np.linspace(start, end, int(round((end-start)/interval, 0))+1)
        # Update outer values
        hyperparameter_values[key] = value_list
        if len(value_list) == 1:
            continue
        tensor_parameters_lists.append(value_list)
    
    #print(f'DEBUG: final_tensor before replacement = \n {final_tensor} ')
    
    for _ in range(1, num_ground_truth_samples):

        random_params = []
        random_coords = []
        for ten_coord_list in tensor_parameters_lists:
            #print(f'DEBUG: ten_coord_list = \n {ten_coord_list}')
            rand_val_tmp = random.choice(ten_coord_list)
            random_params.append(rand_val_tmp)
            random_coords.append(np.where(ten_coord_list == rand_val_tmp)[0][0])

        
        #print(f'DEBUG: random_coords = \n {random_coords}')

        
        evaluation_mode = 'prediction'
        current_hyperparameter_values = indexToHyperparameter(random_coords, hyperparameter_values)
        #print(f'DEBUG: in TENSOR COMPLETION: current_hyperparameter_values = \n {current_hyperparameter_values}')
        true_value_at_random_coord = eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)

        #print(f'DEBUG: final_tensor[random_coords] = \n {final_tensor[tuple(random_coords)]}') 
        #print(f'DEBUG: true_value_at_random_coord = \n {true_value_at_random_coord}') 

        # #print(f'DEBUG: replacing infered values with true values')

        #print(f'DEBUG: random_coords = {random_coords}')
        final_tensor[tuple(random_coords)] = true_value_at_random_coord; 
    #print(f'DEBUG: final_tensor after replacement = \n {final_tensor} ')


    return final_tensor
