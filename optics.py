import numpy as np
import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def np_circ_filter(batch, num_channels, res_h, res_w, filter_radius):
    """create a circular low pass filter
    """
    y,x = np.meshgrid(np.linspace(-(res_w-1)/2, (res_w-1)/2, res_w), np.linspace(-(res_h-1)/2, (res_h-1)/2, res_h))
    mask = x**2+y**2 <= filter_radius**2
    np_filter = np.zeros((res_h, res_w))
    np_filter[mask] = 1.0
    np_filter = np.tile(np.reshape(np_filter, [1,1,res_h,res_w]), [batch, num_channels, 1, 1])
    torch.Tensor(np_filter).to(device)
    return np_filter
    # circ_filter = np_circ_filter(cpx.shape[0], cpx.shape[1], cpx.shape[2], cpx.shape[3], np.min([cpx.shape[2],cpx.shape[3]])/2)
    # cpx_phs = cpx_phs * torch.Tensor(circ_filter).to(device)


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
    phs_only = phs_only[:, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]

    # Move tensors to the same device (CPU or GPU)
    # Depth-to-Space operation
    cpx_phs = torch.nn.functional.pixel_shuffle(phs_only, 2)
    # output = torch.angle(cpx_phs)
    return cpx_phs, amp_max

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


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]
    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted


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

def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


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


def propogation(cpx_in, z, channel, forward=True, inf=True):
    if inf and forward:
        cpx = fftshift(torch.fft.ifftn(cpx_in, dim=(-2, -1), norm='ortho'))
    if inf and not forward:
        cpx = torch.fft.fftn(ifftshift(cpx_in), dim=(-2, -1), norm='ortho')
    return cpx
