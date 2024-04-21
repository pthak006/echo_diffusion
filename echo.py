import numpy as np
import torch
import torch.nn.functional as F


# def create_permutation_matrix(batch_size, exclude_self=True):
#     """Generates a permutation matrix for batch operations, excluding self-reference."""
#     permutation_matrix = []
#     for i in range(batch_size):
#         indices = list(range(batch_size))
#         if exclude_self:
#             indices.remove(i)  # Remove the current index to avoid self-reference
#         torch.manual_seed(0)  # For reproducibility
#         indices = torch.tensor(indices)[torch.randperm(len(indices))]
#         permutation_matrix.append(indices)
#     return torch.stack(permutation_matrix)


# def echo_sample(inputs):
#     f_x, S_x = inputs

#     batch_size, _, dim1, dim2 = f_x.shape
#     d_max = batch_size - 1  # since we exclude self

#     # Create permutation matrix
#     permutation_matrix = create_permutation_matrix(batch_size)

#     # Initialize epsilon and S_cumulative
#     epsilon = torch.zeros_like(f_x)
#     S_cumulative = torch.ones_like(f_x)

#     # Calculate epsilon and S_cumulative for each batch item independently
#     for i in range(batch_size):
#         # Sequential operations within each i, processed independently for each i
#         local_S_cumulative = S_cumulative[i]
#         local_epsilon = epsilon[i]
#         for j in range(d_max):
#             idx = permutation_matrix[i, j]
#             local_epsilon = local_epsilon + local_S_cumulative * f_x[idx]
#             local_S_cumulative = local_S_cumulative * S_x[idx]
#         epsilon[i] = local_epsilon
#         S_cumulative[i] = local_S_cumulative

#     #return epsilon
#     return epsilon

def create_permutation_matrix(batch_size, exclude_self=True):
    """Generates a permutation matrix for batch operations, excluding self-reference."""
    permutation_matrix = []
    torch.manual_seed(0)  # For reproducibility
    for i in range(batch_size):
        indices = list(range(batch_size))
        if exclude_self:
            indices.remove(i)  # Remove the current index to avoid self-reference
        indices = torch.tensor(indices)[torch.randperm(len(indices))]
        permutation_matrix.extend(indices)
    return torch.tensor(permutation_matrix)


def echo_sample(inputs):
    f_x, S_x = inputs

    batch_size, _, dim1, dim2 = f_x.shape
    d_max = batch_size - 1  # since we exclude self

    # Create permutation matrix
    permutation_matrix = create_permutation_matrix(batch_size)

    # Use permutation matrix to index and expand f_x and S_x
    f_x_expanded = f_x[permutation_matrix].view(batch_size * d_max, 1, dim1, dim2)
    S_x_expanded = S_x[permutation_matrix].view(batch_size * d_max, 1, dim1, dim2)

    # Initialize epsilon and S_cumulative
    epsilon = torch.zeros_like(f_x)
    S_cumulative = torch.ones_like(f_x)

    # Calculate epsilon and S_cumulative for each batch item independently
    for j in range(d_max):
        epsilon += S_cumulative * f_x_expanded[j::d_max]
        S_cumulative *= S_x_expanded[j::d_max]

    return epsilon
