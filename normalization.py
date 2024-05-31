def standard_score_normalization(input_tensor):
    mean = input_tensor.mean()
    std = input_tensor.std()

    z_scores = (input_tensor - mean) / std
    normalized_tensor = z_scores

    return normalized_tensor, mean, std

def standard_score_denormalizaion(input_tensor, mean, std):
    denormalized_tensor = input_tensor * std + mean
    return denormalized_tensor