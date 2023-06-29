import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import cv2
from PIL import Image
import optics
import visualization
import Loss_function
import matplotlib.pyplot as plt
import argparse
from torchmetrics import StructuralSimilarityIndexMeasure

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='holografic_slm')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--optimizer', default="adam", type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--z', default=0.1, type=float, help='[m]')
parser.add_argument('--wave_length', default=np.asfarray([638 * 1e-9, 520 * 1e-9, 450 * 1e-9]), type=float, help='[m]')
parser.add_argument('--eval', default=False, type=bool)
parser.add_argument('--model', default='conv', type=str, help='[conv, skip_connection, classic, amp_phs]')
parser.add_argument('--loss', default='[TV_loss]', type=str, help='[TV_loss, L1, L2, perceptual_loss, laplacian_kernel, SSIM_loss]')


class ImageDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        self.transform = ToTensor()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        image = Image.open(self.image_path).convert('RGB')
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # image = Image.open(self.image_path)
        tensor_image = self.transform(image)

        return tensor_image


class CNN_DPE_SKIP(nn.Module):
    def __init__(self):
        super(CNN_DPE_SKIP, self).__init__()
        self.LeakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.conv1r = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2r = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv3r = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)

        self.conv1g = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2g = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv3g = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)

        self.conv1b = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv3b = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)

        custom_weights = torch.tensor([1.0])
        self.linear1 = nn.Linear(1, 1, bias=False)
        self.linear1.weight = nn.Parameter(custom_weights)
        self.linear2 = nn.Linear(1, 1, bias=False)
        self.linear2.weight = nn.Parameter(custom_weights)
        self.linear3 = nn.Linear(1, 1, bias=False)
        self.linear3.weight = nn.Parameter(custom_weights)

        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1r.weight)
        nn.init.xavier_uniform_(self.conv2r.weight)
        nn.init.xavier_uniform_(self.conv3r.weight)
        nn.init.xavier_uniform_(self.conv1g.weight)
        nn.init.xavier_uniform_(self.conv2g.weight)
        nn.init.xavier_uniform_(self.conv3g.weight)
        nn.init.xavier_uniform_(self.conv1b.weight)
        nn.init.xavier_uniform_(self.conv2b.weight)
        nn.init.xavier_uniform_(self.conv3b.weight)

    def forward(self, r, g, b, s0, s1, s2):
        r1 = self.LeakyReLU(self.conv1r(r))
        r2 = self.LeakyReLU(self.conv2r(r1))
        r3 = self.tanh(self.conv3r(r2))
        g1 = self.LeakyReLU(self.conv1g(g))
        g2 = self.LeakyReLU(self.conv2g(g1))
        g3 = self.tanh(self.conv3g(g2))
        b1 = self.LeakyReLU(self.conv1b(b))
        b2 = self.LeakyReLU(self.conv2b(b1))
        b3 = self.tanh(self.conv3b(b2))
        s0 = self.linear1(s0)
        s1 = self.linear2(s1)
        s2 = self.linear3(s2)
        return r3 + r, g3 + g, b3 + b, s0, s1, s2


class CNN_DPE(nn.Module):
    def __init__(self):
        super(CNN_DPE, self).__init__()
        self.LeakyReLU = nn.LeakyReLU()

        self.conv1r = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2r = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv3r = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
        self.conv4r = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv5r = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # Initialize the convolutional layers with identity
        self.conv1r.weight.data.copy_(Loss_function.conv_identity_filter(3))
        self.conv1r.bias.data.fill_(0)
        self.conv2r.weight.data.copy_(Loss_function.conv_identity_filter(5))
        self.conv2r.bias.data.fill_(0)
        self.conv3r.weight.data.copy_(Loss_function.conv_identity_filter(7))
        self.conv3r.bias.data.fill_(0)
        self.conv4r.weight.data.copy_(Loss_function.conv_identity_filter(5))
        self.conv4r.bias.data.fill_(0)
        self.conv5r.weight.data.copy_(Loss_function.conv_identity_filter(3))
        self.conv5r.bias.data.fill_(0)

        self.conv1g = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2g = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv3g = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
        self.conv4g = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv5g = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # Initialize the convolutional layers with identity
        self.conv1g.weight.data.copy_(Loss_function.conv_identity_filter(3))
        self.conv1g.bias.data.fill_(0)
        self.conv2g.weight.data.copy_(Loss_function.conv_identity_filter(5))
        self.conv2g.bias.data.fill_(0)
        self.conv3g.weight.data.copy_(Loss_function.conv_identity_filter(7))
        self.conv3g.bias.data.fill_(0)
        self.conv4g.weight.data.copy_(Loss_function.conv_identity_filter(5))
        self.conv4g.bias.data.fill_(0)
        self.conv5g.weight.data.copy_(Loss_function.conv_identity_filter(3))
        self.conv5g.bias.data.fill_(0)

        self.conv1b = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv3b = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
        self.conv4b = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv5b = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # Initialize the convolutional layers with identity
        self.conv1b.weight.data.copy_(Loss_function.conv_identity_filter(3))
        self.conv1b.bias.data.fill_(0)
        self.conv2b.weight.data.copy_(Loss_function.conv_identity_filter(5))
        self.conv2b.bias.data.fill_(0)
        self.conv3b.weight.data.copy_(Loss_function.conv_identity_filter(7))
        self.conv3b.bias.data.fill_(0)
        self.conv4b.weight.data.copy_(Loss_function.conv_identity_filter(5))
        self.conv4b.bias.data.fill_(0)
        self.conv5b.weight.data.copy_(Loss_function.conv_identity_filter(3))
        self.conv5b.bias.data.fill_(0)

        custom_weights = torch.tensor([1.0])
        self.linear1 = nn.Linear(1, 1, bias=False)
        self.linear1.weight = nn.Parameter(custom_weights)
        self.linear2 = nn.Linear(1, 1, bias=False)
        self.linear2.weight = nn.Parameter(custom_weights)
        self.linear3 = nn.Linear(1, 1, bias=False)
        self.linear3.weight = nn.Parameter(custom_weights)

    def forward(self, r, g, b, s0, s1, s2):
        r1 = self.conv1r(r)
        r2 = self.conv2r(r1)
        r3 = self.conv3r(r2)
        r4 = self.conv3r(r3)
        r5 = self.conv3r(r4)
        g1 = self.conv1g(g)
        g2 = self.conv2g(g1)
        g3 = self.conv3g(g2)
        g4 = self.conv3g(g3)
        g5 = self.conv3g(g4)
        b1 = self.conv1b(b)
        b2 = self.conv2b(b1)
        b3 = self.conv3b(b2)
        b4 = self.conv3b(b3)
        b5 = self.conv3b(b4)
        s0 = self.linear1(s0)
        s1 = self.linear2(s1)
        s2 = self.linear3(s2)
        return r5, g5, b5, s0, s1, s2


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1r = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.conv2r = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2)
        self.conv3r = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.conv4r = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv5r = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.conv1g = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.conv2g = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2)
        self.conv3g = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.conv4g = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv5g = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.conv1b = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2)
        self.conv3b = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.conv4b = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv5b = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1r.weight)
        nn.init.xavier_uniform_(self.conv2r.weight)
        nn.init.xavier_uniform_(self.conv3r.weight)
        nn.init.xavier_uniform_(self.conv4r.weight)
        nn.init.xavier_uniform_(self.conv5r.weight)
        nn.init.xavier_uniform_(self.conv1g.weight)
        nn.init.xavier_uniform_(self.conv2g.weight)
        nn.init.xavier_uniform_(self.conv3g.weight)
        nn.init.xavier_uniform_(self.conv4g.weight)
        nn.init.xavier_uniform_(self.conv5g.weight)
        nn.init.xavier_uniform_(self.conv1b.weight)
        nn.init.xavier_uniform_(self.conv2b.weight)
        nn.init.xavier_uniform_(self.conv3b.weight)
        nn.init.xavier_uniform_(self.conv4b.weight)
        nn.init.xavier_uniform_(self.conv5b.weight)
        custom_weights = torch.tensor([1.0])
        self.linear1 = nn.Linear(1, 1, bias=False)
        self.linear1.weight = nn.Parameter(custom_weights)
        self.linear2 = nn.Linear(1, 1, bias=False)
        self.linear2.weight = nn.Parameter(custom_weights)
        self.linear3 = nn.Linear(1, 1, bias=False)
        self.linear3.weight = nn.Parameter(custom_weights)

    def forward(self, r, g, b, s0, s1, s2):
        r1 = self.relu(self.conv1r(r))
        r2 = self.relu(self.conv2r(r1))
        r3 = self.relu(self.conv2r(r2))
        r4 = self.relu(self.conv2r(r3))
        r5 = self.tanh(self.conv3r(r4))*torch.pi
        g1 = self.relu(self.conv1g(g))
        g2 = self.relu(self.conv2g(g1))
        g3 = self.relu(self.conv2g(g2))
        g4 = self.relu(self.conv2g(g3))
        g5 = self.tanh(self.conv3g(g4))*torch.pi
        b1 = self.relu(self.conv1b(b))
        b2 = self.relu(self.conv2b(b1))
        b3 = self.relu(self.conv2b(b2))
        b4 = self.relu(self.conv2b(b3))
        b5 = self.tanh(self.conv3b(b4))*torch.pi
        s0 = self.linear1(s0)
        s1 = self.linear2(s1)
        s2 = self.linear3(s2)
        return r5, g5, b5, s0, s1, s2


# Train the model
def train(model, train_loader, optimizer, args, epoch):
    model.train()
    train_loss = 0.0

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        new_img, scale = run_setup(images, args, model)

        loss = Loss_function.Loss(new_img, images, epoch, args.loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    return avg_loss, scale


# Validate the model
def validate(model, val_loader, args, epoch):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, images in enumerate(val_loader):
            images = images.to(device)
            new_img, scale = run_setup(images, args, model)

            loss = Loss_function.Loss(new_img, images, epoch, args.loss)
            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    return avg_loss, scale


# Test the model
def test(model, test_loader, criterion, args, repo_path):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)
            new_img, scale = run_setup(images, args, model)

            reproduce_img = np.clip(new_img.cpu().numpy()[0].transpose(1, 2, 0), 0, 1)
            plt.imshow(reproduce_img, cmap='gray')
            plt.title("DPE " + args.model + " Z = " + str(args.z) + '[m]')
            # plt.show(block=False)
            cv2.imwrite(repo_path + '/' + args.model + '.png', (cv2.cvtColor(reproduce_img*255, cv2.COLOR_BGR2RGB)))
            loss = criterion(new_img, images)

            test_loss += loss.item()

    avg_loss = test_loss / len(test_loader)
    return avg_loss


def check_prop(test_loader, args, repo_path):
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)
            new_img = torch.zeros_like(images)
            norm_img = torch.zeros_like(images)
            real_s, img_s = torch.real(images), torch.zeros_like(images)
            cpx_start = torch.complex(real_s, img_s)
            for i, c in enumerate(args.wave_length):
                cpx_inf = optics.propogation(cpx_start[:,i,:,:].unsqueeze(1), args.z, c, forward=False)
                dpe_in, amp_max = optics.dpe(cpx_inf)
                real, img = optics.polar_to_rect(torch.ones_like(dpe_in) * (amp_max/2), dpe_in)
                cpx_slm = torch.complex(real, img)

                f_cpx = optics.fftshift(torch.fft.fftn(cpx_slm, dim=(-2, -1), norm='ortho'))
                f_cpx_filter = optics.np_circ_filter(cpx_slm.shape[0],cpx_slm.shape[1], cpx_slm.shape[2], cpx_slm.shape[3])*f_cpx
                cpx_slm_filter = torch.fft.ifftn(optics.ifftshift(f_cpx_filter), dim=(-2, -1), norm='ortho')

                cpx_recon = optics.propogation(cpx_slm_filter, args.z, c, forward=True)
                # cpx_recon = optics.propogation(cpx_inf, args.z, c, forward=True)
                new_img[:,i,:,:] = torch.real(cpx_recon) ** 2 + torch.imag(cpx_recon) ** 2
                # norm_img[:,i,:,:] = optics.norm_img_energy(new_img[:,i,:,:], images[:,i,:,:])
                norm_img[:,i,:,:] = optics.scale_img(new_img[:,i,:,:], images[:,i,:,:])

            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            print(f"SSIM: {ssim(norm_img, images)}")
            criterion = nn.L1Loss()
            print(f"L2: {criterion(norm_img, images)}")

            recon_img = np.clip(norm_img.cpu().numpy()[0].transpose(1, 2, 0), 0, 1)
            plt.imshow(recon_img, cmap='gray')
            plt.title("DPE with no network Z = " + str(args.z) + '[m]')
            # plt.show(block=True)
            cv2.imwrite(repo_path + '/classic.png', (cv2.cvtColor(recon_img*255, cv2.COLOR_BGR2RGB)))


def run_setup(images, args, model):
    real_s, img_s = torch.real(images), torch.zeros_like(images)
    new_img = torch.zeros_like(images)
    dpe_in = torch.zeros_like(images)
    amp_inf = torch.zeros_like(images)
    phs_inf = torch.zeros_like(images)
    amp_max = torch.Tensor([0, 0, 0])
    cpx_start = torch.complex(real_s, img_s)
    for i, c in enumerate(args.wave_length):
        cpx_inf = optics.propogation(cpx_start[:, i, :, :].unsqueeze(1), args.z, c, forward=False)
        if 'amp_phs' != args.model:
            dpe_in[:,i,:,:], amp_max[i] = optics.dpe(cpx_inf)
        else:
            amp_inf[:,i,:,:], phs_inf[:,i,:,:] = optics.rect_to_polar(cpx_inf.real, cpx_inf.imag)
    if 'amp_phs' != args.model:
        phs_r, phs_g, phs_b, s0, s1, s2 = model(dpe_in[:,0,:,:].unsqueeze(1), dpe_in[:,1,:,:].unsqueeze(1), dpe_in[:,2,:,:].unsqueeze(1),
                                                torch.Tensor([1.0]).to(device), torch.Tensor([1.0]).to(device), torch.Tensor([1.0]).to(device))
        scale = torch.cat([s0.view(1), s1.view(1), s2.view(1)])
        phs = torch.cat([phs_r, phs_g, phs_b], dim=1)
    else:
        input_cpx = torch.cat([amp_inf[:,:1,:,:],phs_inf[:,:1,:,:],amp_inf[:,1:2,:,:],phs_inf[:,1:2,:,:],amp_inf[:,2:,:,:],phs_inf[:,2:,:,:]], dim=1)
        phs_r, phs_g, phs_b, s0, s1, s2 = model(input_cpx[:,:2,:,:],input_cpx[:,2:4,:,:],input_cpx[:,4:,:,:],
                                                torch.Tensor([1.0]).to(device), torch.Tensor([1.0]).to(device),torch.Tensor([1.0]).to(device))
        scale = torch.cat([s0.view(1), s1.view(1), s2.view(1)])
        phs = torch.cat([phs_r, phs_g, phs_b], dim=1)

    for i, c in enumerate(args.wave_length):
        if 'amp_phs' != args.model:
            real, img = optics.polar_to_rect(torch.ones_like(phs[:,i,:,:].unsqueeze(1)) * (amp_max[i] / 2), phs[:,i,:,:].unsqueeze(1))
        else:
            real, img = optics.polar_to_rect(torch.ones_like(phs[:,i,:,:].unsqueeze(1)) * scale[i], phs[:,i,:,:].unsqueeze(1))

        cpx_slm = torch.complex(real, img)

        if 'amp_phs' != args.model:
            f_cpx = optics.fftshift(torch.fft.fftn(cpx_slm, dim=(-2, -1), norm='ortho'))
            f_cpx_filter = optics.np_circ_filter(cpx_slm.shape[0], cpx_slm.shape[1], cpx_slm.shape[2], cpx_slm.shape[3]) * f_cpx
            cpx_slm_filter = torch.fft.ifftn(optics.ifftshift(f_cpx_filter), dim=(-2, -1), norm='ortho')
        else:
            cpx_slm_filter = cpx_slm

        cpx_recon = optics.propogation(cpx_slm_filter, args.z, c, forward=True)
        if 'amp_phs' == args.model:
            new_img[:, i, :, :] = optics.scale_img(torch.real(cpx_recon) ** 2 + torch.imag(cpx_recon) ** 2, images[:,i,:,:])
        else:
            new_img[:, i, :, :] = optics.scale_img(torch.real(cpx_recon) ** 2 + torch.imag(cpx_recon) ** 2, images[:,i,:,:], scale[i])
        # norm_img[:, i, :, :] = optics.scale_img(new_img[:, i, :, :], images[:, i, :, :])
    return new_img, scale


def prep_data(args):
    torch.manual_seed(42)
    # Load the CIFAR-10 dataset
    transform = ToTensor()
    # cifar_dataset = CIFAR10(root="./datasets", train=True, download=True, transform=transform)

    # Specify the path to the image file
    image_path = "./datasets/1.png"
    # Create an instance of the ImageDataset
    dataset = ImageDataset(image_path)
    # Access the image data from the dataset
    # Split the dataset into train, validation, and test sets
    # train_ratio = 0.7
    # val_ratio = 0.2
    # test_ratio = 0.1
    # total_samples = len(dataset)
    # train_size = int(train_ratio * total_samples)
    # val_size = int(val_ratio * total_samples)
    # test_size = total_samples - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(
    #     dataset,
    #     [train_size, val_size, test_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    image_data = dataset
    train_dataset = image_data
    val_dataset = image_data
    test_dataset = image_data

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main():
    # Set random seeds for reproducibility
    args = parser.parse_args()
    repo_path = "./results/prop_dist_" + str(args.z*100).split('.')[0] + "cm"
    if not os.path.isdir(repo_path):
        os.mkdir(repo_path)
    train_loader, val_loader, test_loader = prep_data(args)
    if args.model == 'classic':
        check_prop(test_loader, args, repo_path)
        return
    # Create an instance of the CNN model
    if args.model == 'amp_phs':
        model = CNN().to(device)
        if os.path.isfile(os.path.join(repo_path, args.model + ".pt")):
            model.load_state_dict(torch.load(os.path.join(repo_path, args.model + ".pt")))
            print(f"Load model from {os.path.join(repo_path, args.model + '.pt')}")
    elif args.model == 'conv':
        model = CNN_DPE().to(device)
        if os.path.isfile(os.path.join(repo_path, args.model + ".pt")):
            model.load_state_dict(torch.load(os.path.join(repo_path, args.model + ".pt")))
            print(f"Load model from {os.path.join(repo_path, args.model + '.pt')}")
    elif args.model == 'skip_connection':
        model = CNN_DPE_SKIP().to(device)
        if os.path.isfile(os.path.join(repo_path, args.model + ".pt")):
            model.load_state_dict(torch.load(os.path.join(repo_path, args.model + ".pt")))
            print(f"Load model from {os.path.join(repo_path, args.model + '.pt')}")
    else:
        print("NO PHASE MODEL WAS CHOSEN")
        return
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    if not args.eval:
        num_epochs = args.epochs
        best_val_loss = float('inf')
        train_loss_list = []
        val_loss_list = []
        for epoch in range(num_epochs):
            train_loss, scale = train(model, train_loader, optimizer, args, epoch)
            train_loss_list.append(train_loss)
            # val_loss, scale = validate(model, val_loader, args, epoch)
            # val_loss_list.append(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}")
            # print(f"Train Loss: {train_loss:.4f}, Scale: {scale.cpu().numpy()}")
            # print(f"Validation Loss: {val_loss:.4f}")

            # Save the model if it has the best train loss so far
            if train_loss < best_val_loss:
                torch.save(model.state_dict(), os.path.join(repo_path, args.model + ".pt"))
                best_val_loss = train_loss
                print("Saved the model with the best train loss.")
            # # Save the model if it has the best validation loss so far
            # if val_loss < best_val_loss:
            #     torch.save(model.state_dict(), os.path.join(repo_path, args.model + ".pt"))
            #     best_val_loss = val_loss
            #     print("Saved the model with the best validation loss.")

        # Load the best model for testing
        model.load_state_dict(torch.load(os.path.join(repo_path, args.model + ".pt")))
        data = {'Train loss': train_loss_list, 'Val loss': val_loss_list}
        with open(f'{os.path.join(repo_path, args.model)}.pickle', 'wb') as file:
            # Dump the data into the pickle file
            pickle.dump(data, file)
        visualization.loss_graph(f'{os.path.join(repo_path, args.model)}')


    model.load_state_dict(torch.load(os.path.join(repo_path, args.model + ".pt")))
    # Evaluate the model on the test set
    test_loss = test(model, test_loader, criterion, args, repo_path)
    print(f"Test L2 Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
