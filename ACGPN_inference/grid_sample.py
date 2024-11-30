# !pip install torch  (Install PyTorch)
import torch.nn.functional as F  # Import PyTorch's functional library for advanced operations like grid sampling
from torch.autograd import Variable  # Import Variable for creating tensors with autograd (used in older PyTorch versions)

def grid_sample(input, grid, canvas=None):
    # Perform grid sampling on the input tensor based on the grid
    output = F.grid_sample(input, grid)  # Resamples the input tensor using the coordinates provided in the grid

    # If no canvas is provided, simply return the output
    if canvas is None:
        return output
    else:
        # Create a binary mask of the same size as the input tensor, filled with ones
        # This mask will later help identify areas affected by grid sampling
        input_mask = Variable(input.data.new(input.size()).fill_(1))

        # Apply grid sampling to the mask using the same grid
        output_mask = F.grid_sample(input_mask, grid)
        # The output_mask is a binary tensor indicating regions affected by sampling

        # Blend the grid-sampled output with the canvas
        # - 'output * output_mask': Keeps the resampled areas of the output
        # - 'canvas * (1 - output_mask)': Keeps the unaffected areas of the canvas
        padded_output = output * output_mask + canvas * (1 - output_mask)

        # Return the blended output tensor
        return padded_output
