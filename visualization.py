import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_graph(file_path):
    with open(file_path + '.pickle', 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)

    plt.figure()
    plt.plot(data['Train loss'], label='Train loss')
    plt.plot(data['Val loss'], label='Val loss')
    plt.title('Loss')
    plt.grid()
    plt.legend()
    # plt.show(block=False)
    plt.savefig(f'{file_path}_loss.png')


def calculate_accuracy(model, dataloader, device):
    model.eval() # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([7,7], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix

if __name__ == '__main__':
    dir = '2023_06_17_17_19_08'
    path = "./results"
    loss_graph(f'results/{dir}')
