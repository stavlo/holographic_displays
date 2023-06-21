import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dpe(cpx):
    """
    Anti-aliasing double phase method
    """
    amp = torch.abs(cpx)
    phs = torch.angle(cpx)
    amp_max = torch.max(amp) + 1e-6
    amp = amp / amp_max
    # center phase for each color channel
    phs_zero_mean = phs - phs.mean(dim=(2, 3), keepdim=True)
    # compute two phase maps
    phs_offset = torch.acos(amp)
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset
    # arrange in checkerboard pattern
    phs_1_1 = phs_low[: ,: ,0::2 ,0::2]
    phs_1_2 = phs_high[: ,: ,0::2 ,1::2]
    phs_2_1 = phs_high[: ,: ,1::2 ,0::2]
    phs_2_2 = phs_low[: ,: ,1::2 ,1::2]

    # Concatenating the sliced tensors along the channel dimension
    phs_only = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
    # Move tensors to the same device (CPU or GPU)
    # Depth-to-Space operation
    phs_only = phs_only.view(phs_only.size(0), -1, phs_only.size(2), phs_only.size(3))
    output = torch.nn.functional.pixel_shuffle(phs_only, 2)

    return output

