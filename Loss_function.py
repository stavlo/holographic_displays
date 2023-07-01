import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import models
from collections import namedtuple


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


def histogram_loss(image_path1, image_path2):
    # # Load the image
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    histogram1 = cv2.calcHist(image1, [2], None, [256], [0, 256])
    histogram2 = cv2.calcHist(image2, [2], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Pixel Count")
    plt.plot(histogram1, label='out')
    plt.plot(histogram2, label='in')
    # plt.plot(histogram, label='diff')
    plt.legend()
    plt.xlim([0, 256])
    plt.show()
    #
    # cv2.imshow('conv',image1[:,:,0])
    # cv2.imshow('orig',image2[:,:,0])
    # cv2.waitKey(0)


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


def Loss(img, target, epoch, losses):
    loss = 0
    criterion = torch.nn.MSELoss()
    if 'TV_loss' in losses:
        loss += L1_loss_by_color(img, target, criterion)
        loss += compute_tv_loss(img, target, criterion) * 5e-7
    if 'L2' in losses:
        loss += L1_loss_by_color(img, target, criterion)
    if 'L1' in losses:
        criterion = torch.nn.L1Loss()
        loss += L1_loss_by_color(img, target, criterion)
    if 'laplacian_kernel' in losses:
        loss += laplacian_loss(img, target, criterion)
    if 'SSIM_loss' in losses:
        loss += SSIM_loss(img, target)
    if 'perceptual_loss' in losses:
        model_loss = Vgg16().to(device)
        loss = criterion(model_loss(img).relu3_3, model_loss(target).relu3_3)
    return loss


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        # h = self.slice4(h)
        # h_relu4_3 = h
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)
        return out

if __name__ == "__main__":
    image_path1 = './results/prop_dist_10cm/skip_connection.png'
    image_path1 = './results/prop_dist_10cm/classic.png'
    image_path2 = './datasets/1.png'
    # Load the image
    image1 = torch.Tensor(cv2.imread(image_path1)).view(1,3,1080,1920)
    image2 = torch.Tensor(cv2.imread(image_path2)).view(1,3,1080,1920)

    histogram_loss(image_path1, image_path2)
    compute_tv_loss(image1, image2, torch.nn.L1Loss())