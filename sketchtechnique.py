import numpy as np

from tensorly.tenalg import multi_mode_dot
from generateerrortensor import generateIncompleteErrorTensor
from tensorly import unfold
np.random.seed(1)

def apply_random_projections(tensor, projection_matrices, skipped_index=None):
    """ Apply random projections to each mode of the tensor using multi-mode dot products """
    
    projected_tensor = multi_mode_dot(tensor, projection_matrices, skip=skipped_index)

    return projected_tensor

def tensorCompletionSketchingMRP(eval_func, ranges_dict,  metric, eval_trials=1, known_fraction=0.05,assumed_rank = 2):
    #generate a tensor with zeros in all but random sampled values
    sparse_tensor, _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=known_fraction, metric=metric, eval_trials=eval_trials, empty_are = 'gaussian')
    
    full_sketch_projection_dims = sparse_tensor.ndim*[assumed_rank]


    num_modes = sparse_tensor.ndim

    projection_matrices = [np.transpose(np.random.randn(sparse_tensor.shape[mode], full_sketch_projection_dims[mode])) for mode in range(num_modes)]

    full_sketch_projected_tensor = apply_random_projections(tensor=sparse_tensor, projection_matrices=projection_matrices, skipped_index=None)

    small_sketch_projected_tensors = [] #can be removed
    B_matrices = []
    for index_dropped in range(len(full_sketch_projection_dims)):
        small_sketch_projected_tensor = apply_random_projections(tensor=sparse_tensor, projection_matrices=projection_matrices, skipped_index=index_dropped)
        small_sketch_projected_tensors.append(small_sketch_projected_tensor)
        n_mode_matricization_small_sketch_projected_tensor = unfold(tensor=small_sketch_projected_tensor, mode=index_dropped)
        n_mode_matricization_full_sketch_projected_tensor = unfold(tensor=full_sketch_projected_tensor, mode=index_dropped)
        # Calculate the Moore–Penrose pseudo-inverse of the matricized form
        pseudo_inverse_full_sketch_n_mode_matricization = np.linalg.pinv(n_mode_matricization_full_sketch_projected_tensor)
        B_matrix = np.dot(n_mode_matricization_small_sketch_projected_tensor, pseudo_inverse_full_sketch_n_mode_matricization)
        
        B_matrices.append(B_matrix)

    reconstructed_tensor = multi_mode_dot(full_sketch_projected_tensor, B_matrices)

    return reconstructed_tensor