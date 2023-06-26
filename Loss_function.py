import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def signed_ang(angle):
#     """
#     cast all angles into [-pi, pi]
#     """
#     return (angle + math.pi) % (2*math.pi) - math.pi
#
#
# def grad(img, next_pixel=False, sovel=False):
#     if img.shape[1] > 1:
#         permuted = True
#         img = img.permute(1, 0, 2, 3)
#     else:
#         permuted = False
#
#     # set diff kernel
#     if sovel:  # use sovel filter for gradient calculation
#         k_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32) / 8
#         k_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 8
#     else:
#         if next_pixel:  # x_{n+1} - x_n
#             k_x = torch.tensor([[0, -1, 1]], dtype=torch.float32)
#             k_y = torch.tensor([[1], [-1], [0]], dtype=torch.float32)
#         else:  # x_{n} - x_{n-1}
#             k_x = torch.tensor([[-1, 1, 0]], dtype=torch.float32)
#             k_y = torch.tensor([[0], [1], [-1]], dtype=torch.float32)
#
#     # upload to gpu
#     k_x = k_x.to(img.device).unsqueeze(0).unsqueeze(0)
#     k_y = k_y.to(img.device).unsqueeze(0).unsqueeze(0)
#
#     # boundary handling (replicate elements at boundary)
#     img_x = F.pad(img, (1, 1, 0, 0), 'replicate')
#     img_y = F.pad(img, (0, 0, 1, 1), 'replicate')
#
#     # take sign angular difference
#     grad_x = signed_ang(F.conv2d(img_x, k_x))
#     grad_y = signed_ang(F.conv2d(img_y, k_y))
#
#     if permuted:
#         grad_x = grad_x.permute(1, 0, 2, 3)
#         grad_y = grad_y.permute(1, 0, 2, 3)
#
#     return grad_x, grad_y
#
#
# def laplacian(img):
#     # signed angular difference
#     grad_x1, grad_y1 = grad(img, next_pixel=True)  # x_{n+1} - x_{n}
#     grad_x0, grad_y0 = grad(img, next_pixel=False)  # x_{n} - x_{n-1}
#     laplacian_x = grad_x1 - grad_x0  # (x_{n+1} - x_{n}) - (x_{n} - x_{n-1})
#     laplacian_y = grad_y1 - grad_y0
#     return laplacian_x + laplacian_y
#

def conv_identity_filter(size):
    matrix = torch.full((size, size), 0.0)
    center = int(np.floor(size/2))
    matrix[center, center] = 1
    matrix.view(1, 1, size, size)
    return matrix.to(device)


def laplacian_loss(img1, img2, criterion):
    loss = 0
    H = torch.Tensor([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]]).to(device)
    for i in range(img2.shape[1]):
        laplace1 = F.conv2d(img1[:,i,:,:].unsqueeze(1), H, padding=1)
        laplace2 = F.conv2d(img2[:,i,:,:].unsqueeze(1), H, padding=1)
        loss += criterion(laplace2, laplace1)
    return loss