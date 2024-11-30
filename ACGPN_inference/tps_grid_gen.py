import torch
import itertools
import torch.nn as nn
from torch.autograd import Function, Variable

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
# Computes the partial representation matrix for thin plate splines (TPS)
def compute_partial_repr(input_points, control_points):
    # Get the number of input points and control points
    N = input_points.size(0)
    M = control_points.size(0)

    # Compute pairwise differences between input and control points
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)

    # Element-wise square of pairwise differences (avoids using sum to improve performance)
    pairwise_diff_square = pairwise_diff * pairwise_diff

    # Compute pairwise distances as the sum of squared differences
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]

    # Apply the TPS kernel function: r^2 * log(r), scaled by 0.5
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)

    # Handle numerical errors: replace NaN (from 0 * log(0)) with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)

    return repr_matrix

# Define the TPS (Thin Plate Spline) Grid Generator
class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        # Ensure the target control points are in the correct format
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)  # Number of control points

        self.num_points = N
        target_control_points = target_control_points.float()  # Convert to float for computations

        # Step 1: Create padded kernel matrix for TPS transformation
        forward_kernel = torch.zeros(N + 3, N + 3)  # Initialize the kernel with extra rows/columns for padding
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)

        # Fill the kernel matrix with partial representations and constraints
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)  # Last column set to 1
        forward_kernel[-3, :N].fill_(1)  # Last row set to 1
        forward_kernel[:N, -2:].copy_(target_control_points)  # Add target control points
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))  # Add transposed control points

        # Compute the inverse of the kernel matrix (used for TPS mapping)
        inverse_kernel = torch.inverse(forward_kernel)

        # Step 2: Create a grid of target coordinates (image pixels)
        HW = target_height * target_width  # Total number of pixels
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))  # Cartesian product

        target_coordinate = torch.Tensor(target_coordinate)  # Convert to a tensor (HW x 2)
        Y, X = target_coordinate.split(1, dim=1)  # Split into Y and X coordinates

        # Normalize the coordinates to the range [-1, 1]
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # Combine normalized X and Y into (x, y) format

        # Compute partial representation of the target coordinates
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)

        # Add bias and coordinate terms to the representation
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)

        # Step 3: Register precomputed matrices as buffers for the module
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        # Ensure the input source control points are in the correct format
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2

        batch_size = source_control_points.size(0)  # Number of samples in the batch

        # Concatenate padding matrix to source control points
        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)

        # Compute the mapping matrix by multiplying the inverse kernel with the extended source points
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)

        # Compute source coordinates for each target coordinate
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)

        return source_coordinate
