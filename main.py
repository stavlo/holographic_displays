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
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='holografic_slm')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--optimizer', default="adam", type=str)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--z', default=0.1, type=float, help='[m]')
parser.add_argument('--wave_length', default=np.asfarray([638 * 1e-9, 520 * 1e-9, 450 * 1e-9]), type=float, help='[m]')
parser.add_argument('--check_prop', default=False, type=bool)

class ImageDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        self.transform = ToTensor()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # image = Image.open(self.image_path).convert('RGB')
        image = Image.open(self.image_path)
        tensor_image = self.transform(image)

        return tensor_image


# Define the CNN model
class CNN_DPE(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU()

        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU()

        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# Train the model
def train(model, train_loader, criterion, optimizer, args):
    model.train()
    train_loss = 0.0

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        new_img = run_setup(images, args, model)

        loss = criterion(new_img, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    return avg_loss


# Validate the model
def validate(model, val_loader, criterion, args):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, images in enumerate(val_loader):
            images = images.to(device)
            new_img = run_setup(images, args, model)

            loss = criterion(new_img, images)

            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    return avg_loss


# Test the model
def test(model, test_loader, criterion, args):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)
            new_img = run_setup(images, args, model)

            reproduce_img = np.clip(new_img.cpu().numpy()[0].transpose(1, 2, 0), 0, 1)
            plt.imshow(reproduce_img, cmap='gray')
            plt.show(block=True)
            cv2.imwrite('./results/reproduce_img.png', reproduce_img)
            # cv2.imwrite('./results/reproduce_img.png', (cv2.cvtColor(reproduce_img, cv2.COLOR_BGR2RGB)))
            loss = criterion(new_img, images)

            test_loss += loss.item()

    avg_loss = test_loss / len(test_loader)
    return avg_loss


def check_prop(test_loader, args):
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)
            real_s, img_s = torch.real(images), torch.zeros_like(images)
            cpx_start = torch.complex(real_s, img_s)
            cpx_inf = optics.propogation(cpx_start, args.z, args.wave_length, forward=False, inf=True)
            # dpe_in, amp_max = optics.dpe(cpx_inf)
            # real, img = optics.polar_to_rect(torch.ones_like(dpe_in) * (amp_max/2), dpe_in)
            # real, img = optics.polar_to_rect(torch.ones_like(dpe_in), dpe_in)
            # cpx_slm = torch.complex(real, img)
            # cpx_recon = optics.propogation(cpx_slm, args.z, args.wave_length, forward=True, inf=True)
            cpx_recon = optics.propogation(cpx_inf, args.z, args.wave_length, forward=True, inf=True)
            new_img = torch.real(cpx_recon) ** 2 + torch.imag(cpx_recon) ** 2

            recon_img = np.clip(new_img.cpu().numpy()[0].transpose(1, 2, 0), 0, 1)
            plt.imshow(recon_img, cmap='gray')
            plt.show(block=True)
            cv2.imwrite('./results/check_prop_dpe.png', (cv2.cvtColor(recon_img*255, cv2.COLOR_BGR2RGB)))


def run_setup(images, args, model):
    real_s, img_s = torch.real(images), torch.zeros_like(images)
    cpx_start = torch.complex(real_s, img_s)
    cpx_inf = optics.propogation(cpx_start, args.z, args.wave_length, forward=False, inf=True)

    # # DPE
    # dpe_in, amp_max = optics.dpe(cpx_inf)
    # phs = model(dpe_in)
    # real, img = optics.polar_to_rect(torch.ones_like(phs), phs)
    # cpx_slm = torch.complex(real, img)
    # cpx_recon = optics.propogation(cpx_slm, args.z, args.wave_length, forward=True, inf=True)

    # without DPE
    amp, phs = optics.rect_to_polar(torch.real(cpx_inf), torch.imag(cpx_inf))
    cpx_in = torch.cat([amp, phs], dim=1)
    phs = model(cpx_in)
    real, img = optics.polar_to_rect(torch.ones_like(phs)*0.2, phs)
    cpx_slm = torch.complex(real, img)
    cpx_recon = optics.propogation(cpx_slm, args.z, args.wave_length, forward=True, inf=True)

    new_img = torch.real(cpx_recon) ** 2 + torch.imag(cpx_recon) ** 2
    return new_img

def prep_data(args):
    torch.manual_seed(42)
    # Load the CIFAR-10 dataset
    transform = ToTensor()
    # cifar_dataset = CIFAR10(root="./datasets", train=True, download=True, transform=transform)

    # Specify the path to the image file
    image_path = "./datasets/img.png"
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
    train_loader, val_loader, test_loader = prep_data(args)
    if args.check_prop:
        check_prop(test_loader, args)
        return
    # Create an instance of the CNN model
    model = CNN().to(device)
    # Define the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Training loop

    num_epochs = args.epochs
    best_val_loss = float('inf')
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, args)
        train_loss_list.append(train_loss)
        val_loss = validate(model, val_loader, criterion, args)
        val_loss_list.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), "results/best_model.pt")
            best_val_loss = val_loss
            print("Saved the model with the best validation loss.")

    # Load the best model for testing
    model.load_state_dict(torch.load("results/best_model.pt"))
    data = {'Train loss': train_loss_list, 'Val loss': val_loss_list}
    with open(f'./results/loss.pickle', 'wb') as file:
        # Dump the data into the pickle file
        pickle.dump(data, file)
    visualization.loss_graph(f'./results/')


    # Evaluate the model on the test set
    test_loss = test(model, test_loader, criterion, args)
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
