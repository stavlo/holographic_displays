import numpy as np
import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def np_circ_filter(batch, num_channels, res_h, res_w, filter_radius=None):
    """create a circular low pass filter
    """
    if filter_radius == None:
        filter_radius = int(np.min([res_h, res_w]) / 2)
    y,x = np.meshgrid(np.linspace(-(res_w-1)/2, (res_w-1)/2, res_w), np.linspace(-(res_h-1)/2, (res_h-1)/2, res_h))
    mask = x**2+y**2 <= filter_radius**2
    np_filter = np.zeros((res_h, res_w))
    np_filter[mask] = 1.0
    np_filter = np.tile(np.reshape(np_filter, [1,1,res_h,res_w]), [batch, num_channels, 1, 1])
    np_filter = torch.Tensor(np_filter).to(device)
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
    phs_zero_mean = phs - phs.mean(dim=(-2, -1), keepdim=True)
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
    # phs_only = phs_only[:, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]

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


def propogation(cpx_in, z, wave_length, forward=True):
    # do we need scale ???
    # scale = np.sqrt(2*np.pi) # cpx_in.shape[-1]*cpx_in.shape[-2]
    # to image plane
    if forward:
        prop_phs = prop_mask(cpx_in, z, wave_length)
        cpx = torch.fft.fftn(ifftshift(cpx_in), dim=(-2, -1), norm='ortho') * prop_phs
        cpx = fftshift(torch.fft.ifftn(cpx, dim=(-2, -1), norm='ortho'))
    # to source plane
    if not forward:
        prop_phs = prop_mask(cpx_in, -z, wave_length)
        cpx = torch.fft.fftn(ifftshift(cpx_in), dim=(-2, -1), norm='ortho') * prop_phs
        cpx = fftshift(torch.fft.ifftn(cpx, dim=(-2, -1), norm='ortho'))
    return cpx


def prop_mask(cpx_in, z, wave_length):
    # resolution of input field, should be: (num_images, num_channels, height, width, 2)
    field_resolution = cpx_in.size()
    # number of pixels
    num_y, num_x = field_resolution[-2], field_resolution[-1]
    # sampling interval size
    feature_size = (6.4 * 1e-6, 6.4 * 1e-6)
    dy, dx = feature_size
    # size of the field
    y, x = (dy * float(num_y), dx * float(num_x))
    # frequency coordinates sampling
    fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
    fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)
    # momentum/reciprocal space
    FX, FY = np.meshgrid(fx, fy)
    # transfer function in numpy (omit distance)
    HH = 2 * math.pi * np.sqrt(1 / wave_length**2 - (FX**2 + FY**2))
    # create tensor & upload to device (GPU)
    H_exp = torch.tensor(HH, dtype=torch.float32).to(cpx_in.device)
    # reshape tensor and multiply
    H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))
    # multiply by distance
    H_exp = torch.mul(H_exp, z)
    # band-limited ASM - Matsushima et al. (2009)
    fy_max = 1 / np.sqrt((2 * z * (1 / y)) ** 2 + 1) / wave_length
    fx_max = 1 / np.sqrt((2 * z * (1 / x)) ** 2 + 1) / wave_length
    H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=torch.float32)
    # get real/img components
    H_real, H_imag = polar_to_rect(H_filter.to(cpx_in.device), H_exp)
    H = torch.stack((H_real, H_imag), 4)
    H = ifftshift(H)
    H = torch.view_as_complex(H)
    return H


def norm_img_energy(img1, original):
    energy1 = torch.sum(torch.sqrt(img1 ** 2))
    orignal_e = torch.sum(torch.sqrt(original ** 2))
    ratio = orignal_e / energy1
    norm_img = img1 * ratio
    return norm_img


def scale_img(img1, original, net_s=1):
    img1 = img1 * net_s
    s = (img1 * original).mean() / (img1 ** 2).mean()  # scale minimizing MSE btw recon and    return scale
    return s * img1
