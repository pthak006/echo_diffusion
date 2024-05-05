import numpy as np
import torch
import torch.nn.functional as F


def create_permutation_matrix(batch_size, exclude_self=True):
    """Generates a permutation matrix for batch operations, excluding self-reference."""
    permutation_matrix = []
    for i in range(batch_size):
        indices = list(range(batch_size))
        if exclude_self:
            indices.remove(i)  # Remove the current index to avoid self-reference
        torch.manual_seed(0)  # For reproducibility
        indices = torch.tensor(indices)[torch.randperm(len(indices))]
        permutation_matrix.append(indices)
    return torch.stack(permutation_matrix)


def echo_sample(inputs):
    f_x, S_x = inputs

    batch_size, _, dim1, dim2 = f_x.shape
    d_max = batch_size - 1  # since we exclude self

    # Create permutation matrix
    permutation_matrix = create_permutation_matrix(batch_size)

    # Initialize epsilon and S_cumulative
    epsilon = torch.zeros_like(f_x)
    S_cumulative = torch.ones_like(f_x)

    # Calculate epsilon and S_cumulative for each batch item independently
    for i in range(batch_size):
        # Sequential operations within each i, processed independently for each i
        local_S_cumulative = S_cumulative[i]
        local_epsilon = epsilon[i]
        for j in range(d_max):
            idx = permutation_matrix[i, j]
            local_epsilon = local_epsilon + local_S_cumulative * f_x[idx]
            local_S_cumulative = local_S_cumulative * S_x[idx]
        epsilon[i] = local_epsilon
        S_cumulative[i] = local_S_cumulative

    #return epsilon
    return epsilon


def echo_loss(S):
    # Define the scaling factor
    scaling_factor = 10

    # Get the batch size and dimensions of the matrix S
    batch_size, _, dim, dim = S.shape  # Reusing the variable 'dim' for clarity in dimensions

    # Scale the matrices S by the scaling factor
    scaled_S = scaling_factor * S

    # Calculate the scaled log determinant
    _, scaled_log_abs_det = torch.linalg.slogdet(scaled_S)

    # Calculate the mean of scaled log determinants
    scaled_mi_loss = torch.mean(scaled_log_abs_det)

    # Adjust the scaled mean log determinant by subtracting dim*log(scaling_factor)
    mi_loss = scaled_mi_loss - (dim * torch.log(torch.tensor(scaling_factor, device=S.device)))

    # Return the calculated mi_loss
    return mi_loss


# def create_permutation_matrix(batch_size, exclude_self=True):
#     """Generates a permutation matrix for batch operations, excluding self-reference."""
#     permutation_matrix = []
#     torch.manual_seed(0)  # For reproducibility
#     for i in range(batch_size):
#         indices = list(range(batch_size))
#         if exclude_self:
#             indices.remove(i)  # Remove the current index to avoid self-reference
#         indices = torch.tensor(indices)[torch.randperm(len(indices))]
#         permutation_matrix.extend(indices)
#     return torch.tensor(permutation_matrix)


# def echo_sample(inputs):
#     f_x, S_x = inputs

#     batch_size, _, dim1, dim2 = f_x.shape
#     d_max = batch_size - 1  # since we exclude self

#     # Create permutation matrix
#     permutation_matrix = create_permutation_matrix(batch_size)

#     # Use permutation matrix to index and expand f_x and S_x
#     f_x_expanded = f_x[permutation_matrix].view(batch_size * d_max, 1, dim1, dim2)
#     S_x_expanded = S_x[permutation_matrix].view(batch_size * d_max, 1, dim1, dim2)

#     # Initialize epsilon and S_cumulative
#     epsilon = torch.zeros_like(f_x)
#     S_cumulative = torch.ones_like(f_x)

#     # Calculate epsilon and S_cumulative for each batch item independently
#     for j in range(d_max):
#         epsilon += S_cumulative * f_x_expanded[j::d_max]
#         S_cumulative *= S_x_expanded[j::d_max]

#     return epsilon
