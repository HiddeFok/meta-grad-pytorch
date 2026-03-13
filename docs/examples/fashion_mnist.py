import os
import random
import time
from datetime import timedelta
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adagrad, Adam

from parameterfree.kt import KT
from parameterfree.ckt import cKT
from parameterfree.cocob import COCOB

from metagrad import (
    CoordinateMetaGrad,
    FullBlockMetagrad,
    FullMetaGrad,
    SketchedBlockMetaGrad,
    SketchedMetaGrad,
)


# Copied and adapted from https://github.com/bremen79/parameterfree/blob/main/examples/test_on_fashionmnist.py
class FashionMNISTModel(nn.Module):
    def __init__(self, hidden_dim=1000):
        super(FashionMNISTModel, self).__init__()

        self.fc1 = nn.Linear(28*28, hidden_dim)  # Input layer (28x28 = 784 pixels)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # First hidden layer
        self.fc3 = nn.Linear(hidden_dim, 10)  # Output layer (10 classes for FashionMNIST)
    
    def forward(self, x):
        x = nn.Flatten()(x)
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device():
    try:
        if torch.backends.mps.is_available():
            print("Using MPS")
            # For small networks and loads this cpu seems to be faster 
            # device = torch.device('cpu')
            device = torch.device('cpu')
        elif torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device('cuda')
        else:
            print("Using CPU")
            device = torch.device('cpu')
    except AttributeError as e:
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device('cuda')
        else:
            print("Using CPU")
            device = torch.device('cpu')
    return device


def load_or_download_data(batch_size=128):
    os.makedirs("./checkpoints/data", exist_ok=True)

    train_data = FashionMNIST(root="./checkpoints/data", train=True, transform=ToTensor(), download=True)
    test_data = FashionMNIST(root="./checkpoints/data", train=False, transform=ToTensor())

    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset


def train_and_eval(model, optimizer, train_dataset, test_dataset, device, max_epochs=30):
    loss_func = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_acc = 0
    total = 0
    num_batch = 0

    train_losses = []
    train_accs = []

    test_losses = []
    test_accs = []

    time_start = time.time()
    for epoch in range(max_epochs):
        for batch_idx, (data, targets) in enumerate(train_dataset):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            logit_preds = model(data)
            loss = loss_func(logit_preds, targets)
            loss.backward()
            optimizer.step()

            total += targets.shape[0]
            total_loss += loss.item()
            _, preds = torch.max(logit_preds, dim=1)
            total_acc += (preds == targets).sum().item()
            num_batch += 1

            if batch_idx % 100 == 0:
                elapsed = timedelta(seconds=time.time() - time_start)

                print("{}, {}, {}, {}, {}, {}".format(
                    f"Epoch {epoch + 1}/{max_epochs}",
                    f"Batch {batch_idx}/{len(train_dataset)}",
                    f"Minibatch Loss: {loss.item():.3f}",
                    f"Online Loss: {total_loss/num_batch:.3f}",
                    f"Online Acc: {total_acc/total:.3f}",
                    f"Time: {str(elapsed)[:-3]}",
                ))

        # Evaluation on  training data
        model.eval()
        total_loss_tr_eval = 0
        total_acc_tr_eval = 0
        total_tr_eval = 0
        num_batch_tr_eval = 0
        for batch_idx, (data, targets) in enumerate(train_dataset):
            data, targets = data.to(device), targets.to(device)

            logit_preds = model(data)
            loss = loss_func(logit_preds, targets)
            _, predicted = torch.max(logit_preds, dim=1)

            total_tr_eval += targets.shape[0]
            total_acc_tr_eval += (predicted == targets).sum().item()
            total_loss_tr_eval += loss.item()
            num_batch_tr_eval += 1

        train_losses.append(total_loss_tr_eval / num_batch_tr_eval)
        train_accs.append(total_acc_tr_eval / total_tr_eval)

        # Evaluation on test data
        total_loss_te_eval = 0
        total_acc_te_eval = 0
        total_te_eval = 0
        num_batch_te_eval = 0
        for batch_idx, (data, targets) in enumerate(test_dataset):
            data, targets = data.to(device), targets.to(device)

            logit_preds = model(data)
            loss = loss_func(logit_preds, targets)
            _, predicted = torch.max(logit_preds, dim=1)

            total_te_eval += targets.shape[0]
            total_acc_te_eval += (predicted == targets).sum().item()
            total_loss_te_eval += loss.item()
            num_batch_te_eval += 1

        test_losses.append(total_loss_te_eval / num_batch_te_eval)
        test_accs.append(total_acc_te_eval / total_te_eval)


    res = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
    }
    return res


if __name__ == "__main__":
    MAX_EPOCHS = 30
    BATCH_SIZE = 100
    SEED = 123
    CKPT_DIR = "./checkpoints"
    
    set_seed(SEED)
    device = get_device()

    model = FashionMNISTModel(hidden_dim=100)
    model.to(device)
    # optimizer = SketchedMetaGrad(model.parameters(), sigma=0.5, D_inf=3, sketch_size=5)
    # optimizer = CoordinateMetaGrad(model.parameters(), sigma=0.5, D_inf=3)
    # optimizer = Adam(params=model.parameters(), lr=0.01)
    # optimizer = COCOB(model.parameters())
    optimizer = KT(model.parameters())
    train_dataset, test_dataset = load_or_download_data(batch_size=BATCH_SIZE)

    train_test_results = train_and_eval(
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        test_dataset=test_dataset,
        device=device, 
        max_epochs=MAX_EPOCHS
    )

    model_dir = f"{CKPT_DIR}/models"
    os.makedirs(model_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{model_dir}/kt.pt")

    with open(f"{model_dir}/kt_metrics.json", "w") as f:
        json.dump(train_test_results, f)
