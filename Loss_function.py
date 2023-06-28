import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
from torchmetrics import StructuralSimilarityIndexMeasure

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        # mask = torch.abs(laplace1) < 0.5
        # loss += torch.sum(torch.abs(laplace1*mask)) * 0.00001
        laplace2 = F.conv2d(img2[:,i,:,:].unsqueeze(1), H, padding=1)
        loss += criterion(laplace2, laplace1)
    return loss


# compute total variation
def compute_tv_4d(field):
    dx = field[:, :, 1:] - field[:, :, :-1]
    dy = field[:, 1:, :] - field[:, :-1, :]
    return dx, dy


# compute total variation loss
def compute_tv_loss(x_in, x_gt, criterion):
    loss = 0
    for i in range(x_in.shape[1]):
        x_in_dx, x_in_dy = compute_tv_4d(x_in[:,i,:,:])
        x_out_dx, x_out_dy = compute_tv_4d(x_gt[:,i,:,:])
        tv_loss = torch.sum(torch.abs(x_in_dx - x_out_dx)) + torch.sum(torch.abs(x_in_dy - x_out_dy))
        loss += tv_loss
    return loss * 10


def histogram_loss():
    image_path1 = './results/prop_dist_50cm/conv.png'
    image_path2 = './datasets/1.png'
    # Load the image
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    histogram = cv2.calcHist([np.abs(image1[:,:,0] - image2[:,:,0])], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Pixel Count")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()

    cv2.imshow('conv',image1[:,:,0])
    cv2.waitKey(0)



def L1_loss_by_color(img, target, criterion):
    loss = 0
    for c in range(3):
        loss += criterion(img[:, c, :, :], target[:, c, :, :])
    return loss


def SSIM_loss(img, target):
    loss = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    for c in range(3):
        loss += 1 - ssim(img[:, c, :, :].unsqueeze(1), target[:, c, :, :].unsqueeze(1))
    return loss


def Loss(img, target, criterion):
    loss = 0
    loss += L1_loss_by_color(img, target, criterion)
    loss += compute_tv_loss(img, target, criterion) * 1e-6
    # loss += laplacian_loss(img, target, criterion)
    # loss += SSIM_loss(img, target)
    return loss

if __name__ == "__main__":
    image_path1 = './results/prop_dist_50cm/conv.png'
    image_path2 = './datasets/1.png'
    # Load the image
    image1 = torch.Tensor(cv2.imread(image_path1)).view(1,3,1080,1920)
    image2 = torch.Tensor(cv2.imread(image_path2)).view(1,3,1080,1920)
    compute_tv_loss(image1, image2, torch.nn.L1Loss())