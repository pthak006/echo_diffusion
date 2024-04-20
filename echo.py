import numpy as np
import torch
import torch.nn.functional as F


# def random_indices(n, d):
#     # Generates a 1D tensor of random integers from 0 to n-1, with total n*d elements.
#     return torch.randint(low=0, high=n, size=(n * d,), dtype=torch.int32)


# def gather_nd_reshape(t, indices):
#     # Get the shape of the input tensor t
#     t_shape = t.shape

#     # Convert the indices tensor to long type
#     indices = indices.long()

#     # Calculate the flat indices based on the provided indices tensor
#     strides = torch.tensor(
#         [np.prod(t_shape[i + 1 :]) for i in range(len(t_shape) - 1)] + [1]
#     )
#     flat_indices = (indices * strides[-indices.shape[-1] :]).sum(dim=-1)

#     # Gather the values from the flattened tensor
#     gathered = t.view(-1)[flat_indices]

#     return gathered


# def indices_without_replacement(batch_size, d_max=-1, replace=False, pop=True):
#     if d_max < 0:
#         d_max = batch_size + d_max

#     inds = torch.empty((0, d_max, 2), dtype=torch.long)
#     for i in range(batch_size):
#         batch_range = torch.arange(batch_size)
#         if pop:
#             batch_range = batch_range[batch_range != i]
#         shuffled_indices = torch.randperm(batch_range.size(0))[:d_max]
#         dmax_range = torch.arange(d_max)

#         dmax_enumerated = torch.stack(
#             (dmax_range, batch_range[shuffled_indices]), dim=1
#         )
#         inds = torch.cat((inds, dmax_enumerated.unsqueeze(0)), dim=0)


# def permute_neighbor_indices(batch_size, d_max=-1, replace=False, pop=True):
#     if d_max < 0:
#         d_max = batch_size + d_max

#     inds = []
#     for i in range(batch_size):
#         if pop:
#             # Exclude the current sample if pop is True
#             sub_batch = torch.cat((torch.arange(i), torch.arange(i + 1, batch_size)))
#         else:
#             sub_batch = torch.arange(batch_size)

#         if replace:
#             # Select d_max elements with replacement
#             selected_indices = torch.multinomial(
#                 torch.ones(sub_batch.shape), num_samples=d_max, replacement=True
#             )
#             selected_indices = sub_batch[selected_indices]
#         else:
#             # Shuffle the sub_batch and select the first d_max elements
#             selected_indices = sub_batch[torch.randperm(sub_batch.shape[0])[:d_max]]

#         # Pair each selected index with its position in the batch
#         dmax_range = torch.arange(d_max)
#         dmax_enumerated = torch.stack((dmax_range, selected_indices), dim=1)
#         inds.append(dmax_enumerated)

#     # Stack the indices from all batches into a single tensor
#     inds_tensor = torch.stack(inds, dim=0)
#     return inds_tensor


# def echo_sample(
#    inputs,
#    clip=None,
#    d_max=100,
#    batch_size=100,
#    multiplicative=False,
#    echo_mc=False,
#    replace=False,
#    fx_clip=None,
#    plus_sx=True,
#    calc_log=True,
#    set_batch=True,
#    return_noise=False,
# ):
#    if isinstance(inputs, list):
#        fx = inputs[0]
#        sx = inputs[1]
#    else:
#        fx = inputs
#        sx = None
#
#    if clip is None:
#        max_fx = fx_clip if fx_clip is not None else 1.0
#        clip = (2 ** (-23) / max_fx) ** (1.0 / d_max)
#
#    clip = torch.tensor(clip, dtype=fx.dtype, device=fx.device)
#
#    if fx_clip is not None:
#        fx = torch.clamp(fx, -fx_clip, fx_clip)
#
#    if sx is not None:
#        if not calc_log:
#            sx = clip * sx
#            sx = torch.where(
#                torch.abs(sx) < torch.finfo(sx.dtype).eps,
#                torch.sign(sx) * torch.finfo(sx.dtype).eps,
#                sx,
#            )
#        else:
#            sx = torch.log(clip) + (-sx if not plus_sx else sx)
#    else:
#        sx = torch.zeros_like(fx)
#
#    if echo_mc:
#        fx = fx - fx.mean(dim=0, keepdim=True)
#
#    if replace:
#        sx = sx.view(sx.size(0), -1) if len(sx.shape) > 2 else sx
#        fx = fx.view(fx.size(0), -1) if len(fx.shape) > 2 else fx
#
#        inds = torch.randint(
#            0, batch_size, (batch_size * d_max,), dtype=torch.long, device=fx.device
#        )
#        inds = inds.view(-1, 1)
#
#        select_sx = gather_nd_reshape(sx, inds).view(batch_size, d_max, -1)
#        select_fx = gather_nd_reshape(fx, inds).view(batch_size, d_max)
#
#        if len(sx.shape) > 2:
#            select_sx = select_sx.view(batch_size, d_max, *sx.shape[1:])
#            sx = sx.unsqueeze(1)
#
#        if len(fx.shape) > 2:
#            select_fx = select_fx.view(batch_size, d_max, *fx.shape[1:])
#            fx = fx.unsqueeze(1)
#    else:
#        repeat_fx = torch.ones_like(fx.unsqueeze(0)) * torch.ones_like(fx.unsqueeze(1))
#        stack_fx = fx * repeat_fx
#
#        repeat_sx = torch.ones_like(sx.unsqueeze(0)) * torch.ones_like(sx.unsqueeze(1))
#        stack_sx = sx * repeat_sx
#
#        if not set_batch:
#            inds = indices_without_replacement(batch_size, d_max)
#        else:
#            inds = permute_neighbor_indices(batch_size, d_max, replace=replace)
#
#        select_sx = gather_nd_reshape(stack_sx, inds).view(batch_size, d_max, -1)
#        select_fx = gather_nd_reshape(stack_fx, inds).view(batch_size, d_max, -1)
#
#    if calc_log:
#        sx_echoes = torch.cumsum(select_sx, dim=1)
#    else:
#        sx_echoes = torch.cumprod(select_sx, dim=1)
#
#    sx_echoes = torch.exp(sx_echoes) if calc_log else sx_echoes
#
#    fx_sx_echoes = select_fx * sx_echoes
#    noise = torch.sum(fx_sx_echoes, dim=1)
#    print(f"Shape of noise: {noise.shape}")
#    print(f"First element of noise: {noise[0]}")
#
#    if sx is not None:
#        sx = sx if not calc_log else torch.exp(sx)
#
#    noise_expanded = noise.view(batch_size, -1)
#    noise_expanded = noise.view(batch_size, *([1] * (len(sx.shape) - 2)), -1)
#    print(f"Shape of noise_expanded: {noise_expanded.shape}")
#    print(f"First element of noise_expanded: {noise_expanded[0]}")
#
#    if multiplicative:
#        output = torch.exp(fx + sx * noise_expanded)
#    else:
#        output = fx + sx * noise_expanded
#        return output if not return_noise else noise


# def echo_loss(inputs, clip=0.8359, calc_log=True, plus_sx=True, multiplicative=False):
#    if isinstance(inputs, list):
#        z_mean = inputs[0]
#        z_scale = inputs[-1]
#    else:
#        z_scale = inputs
#
#    clip_tensor = torch.tensor(clip, dtype=z_scale.dtype, device=z_scale.device)
#
#    if not calc_log:
#        z_scale_clipped = clip_tensor * z_scale
#        z_scale_clipped = torch.where(
#            torch.abs(z_scale_clipped) < 1e-7,
#            torch.sign(z_scale_clipped) * 1e-7,
#            z_scale_clipped,
#        )
#        mi = -torch.log(torch.abs(z_scale_clipped) + 1e-7)
#    else:
#        if plus_sx:
#            mi = -(torch.log(clip_tensor) + z_scale)
#        else:
#            mi = -(torch.log(clip_tensor) - z_scale)
#
#    averaged_loss = calculate_avg_log_det(mi)
#
#    return averaged_loss
#

# def calculate_avg_log_det(sx_matrices, eps=1e-6):
#    # Add a small diagonal value to ensure positive definiteness
#    sx_matrices = sx_matrices + eps * torch.eye(
#        sx_matrices.size(-1), dtype=sx_matrices.dtype, device=sx_matrices.device
#    )
#
#    # Calculate the negative log-determinant for each Sx matrix in the batch
#    log_det_sx = -torch.logdet(sx_matrices)
#
#    # Replace any infinite or nan values with zero
#    log_det_sx = torch.where(
#        torch.isinf(log_det_sx) | torch.isnan(log_det_sx),
#        torch.zeros_like(log_det_sx),
#        log_det_sx,
#    )
#
#    # Average the negative log-determinants across the batch
#    avg_log_det_sx = torch.mean(log_det_sx)
#
#    return avg_log_det_sx


def echo_loss(z_scale, clip_value=1e-7):
    # Ensure that z_scale has the expected shape [batch_size, 1, dim1, dim2]
    assert (
        len(z_scale.shape) == 4 and z_scale.shape[1] == 1
    ), "z_scale must have shape [batch_size, 1, dim1, dim2]"

    # Clip the diagonal values of S(x) for numerical stability
    z_scale_clipped = torch.clamp(z_scale, min=clip_value)

    # Extract the diagonal values from z_scale_clipped
    diagonal_values = torch.diagonal(z_scale_clipped, dim1=-2, dim2=-1)

    # Calculate the determinant by multiplying the diagonal values
    det_sx = torch.prod(diagonal_values, dim=-1, keepdim=True)

    # Take the absolute value of the determinant
    abs_det_sx = torch.abs(det_sx)

    # Add a small value to the absolute determinant to avoid taking the logarithm of zero
    abs_det_sx_stable = abs_det_sx + torch.finfo(abs_det_sx.dtype).tiny

    # Calculate the negative logarithm of the absolute determinant
    neg_log_abs_det_sx = -torch.log(abs_det_sx_stable)

    # Handle potential nan values
    neg_log_abs_det_sx = torch.where(
        torch.isnan(neg_log_abs_det_sx),
        torch.zeros_like(neg_log_abs_det_sx),
        neg_log_abs_det_sx,
    )

    # Handle potential infinite values
    neg_log_abs_det_sx = torch.where(
        torch.isinf(neg_log_abs_det_sx),
        torch.zeros_like(neg_log_abs_det_sx),
        neg_log_abs_det_sx,
    )

    # Average the negative logarithm of the absolute determinant across the batch
    avg_neg_log_abs_det_sx = torch.mean(neg_log_abs_det_sx)

    return avg_neg_log_abs_det_sx


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

    # Calculate the final output z
    z = f_x + S_x * epsilon
    return z
