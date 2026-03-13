import torch, os
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

from torch.optim import AdamW

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ConditionalFlowMatching



def load_dataset(dataset, dataset_path, train_batch_size, inference_batch_size):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST, CIFAR10

    transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if dataset == 'CIFAR10':
        train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
        test_dataset  = CIFAR10(dataset_path, transform=transform, train=False, download=True)
    else:
        train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
        test_dataset  = MNIST(dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=inference_batch_size, shuffle=False,  **kwargs)

    return train_loader, test_loader


def train_cfm(settings):
    """
    Trains the conditional flow matching model.
    Returns a dictionary of the training results.
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(settings["gpu_id"])
    DEVICE = torch.device("cuda:0".format(settings["gpu_id"]) if settings["cuda"] else "cpu")
    model = ConditionalFlowMatching(image_resolution=settings["img_size"],
                 hidden_dims=settings["hidden_dims"], sigma_min=settings["sigma_min"]).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.99))

    print("Start training CFM...")
    model.train()

    train_loader, test_loader = load_dataset(settings["dataset"], settings["dataset_path"], settings["train_batch_size"], settings["inference_batch_size"])

    img_size = (28, 28, 1)
    hidden_dims = [settings["hidden_dim"] for _ in range(settings["n_layers"])]
    sigma_min = settings["sigma_min"]

    for epoch in range(settings["n_epochs"]):
        total_loss = 0
        for batch_idx, (x_1, _) in enumerate(train_loader):
            optimizer.zero_grad()

            # sample Gaussian normal and target data
            x_1 = x_1.to(DEVICE)
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], 1, 1, 1, device=DEVICE)

            # get noise-interpolated data
            x_t = model.interpolate(x_0, x_1, t)

            # target velocity
            velocity_target = model.get_velocity(x_0, x_1)
            # estimate velocity
            velocity_pred = model(x_t, t)

            # conditional flow matching loss. model learns
            loss = ((velocity_pred - velocity_target) ** 2).mean()
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

            if batch_idx % 100 == 0:
                print("\t\tCFM loss: ", loss.item(), "  grad_norm:", grad_norm)

        print("\tEpoch", epoch + 1, "complete!", "\tCFM Loss: ", total_loss / len(train_loader))

    print("Finish!!")

def test_cfm(settings):
    """
    Tests the conditional flow matching model.
    Returns a dictionary of the test results.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(settings["gpu_id"])
    DEVICE = torch.device("cuda:0".format(settings["gpu_id"]) if settings["cuda"] else "cpu")
    model = ConditionalFlowMatching(image_resolution=settings["img_size"],
                 hidden_dims=settings["hidden_dims"], sigma_min=settings["sigma_min"]).to(DEVICE)
    model.eval()

    test_loader = load_dataset(settings["dataset"], settings["dataset_path"], settings["inference_batch_size"], settings["inference_batch_size"])

    for batch_idx, (x_1, _) in enumerate(test_loader):
        x_1 = x_1.to(DEVICE)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, 1, 1, device=DEVICE)

        x_t = model.interpolate(x_0, x_1, t)
        velocity_pred = model(x_t, t)
        velocity_target = model.get_velocity(x_0, x_1)
    
    return velocity_pred, velocity_target

def main():
    settings = {
        "dataset": "MNIST",
        "dataset_path": "~/datasets",
        "train_batch_size": 128,
        "inference_batch_size": 64,
        "gpu_id": 2,
        "cuda": True,
    }
    train_cfm(settings)
    test_cfm(settings)

if __name__ == "__main__":
    main()