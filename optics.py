import numpy as np
import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dpe(cpx):
    """
    Anti-aliasing double phase method
    """
    amp, phs = rect_to_polar(torch.real(cpx), torch.imag(cpx))
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
    cpx_phs = torch.nn.functional.pixel_shuffle(phs_only, 2)
    output = torch.angle(cpx_phs)
    return output

def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def ifft2(tensor_re, tensor_im, shift=False):
    """Applies a 2D ifft to the complex tensor represented by tensor_re and _im"""
    tensor_out = torch.stack((tensor_re, tensor_im), 4)

    if shift:
        tensor_out = ifftshift(tensor_out)
    (tensor_out_re, tensor_out_im) = torch.ifft(tensor_out, 2, True).split(1, 4)

    tensor_out_re = tensor_out_re.squeeze(4)
    tensor_out_im = tensor_out_im.squeeze(4)

    return tensor_out_re, tensor_out_im


def fft2(tensor_re, tensor_im, shift=False):
    """Applies a 2D fft to the complex tensor represented by tensor_re and _im"""
    # fft2
    (tensor_out_re, tensor_out_im) = torch.fft(torch.stack((tensor_re, tensor_im), 4), 2, True).split(1, 4)

    tensor_out_re = tensor_out_re.squeeze(4)
    tensor_out_im = tensor_out_im.squeeze(4)

    # apply fftshift
    if shift:
        tensor_out_re = fftshift(tensor_out_re)
        tensor_out_im = fftshift(tensor_out_im)

    return tensor_out_re, tensor_out_im


def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def propogation(cpx_in, z, channel, inf=True):
    if inf:
        cpx = torch.fft.fftn(ifftshift(cpx_in), dim=(-2, -1), norm='ortho')
    return cpx
