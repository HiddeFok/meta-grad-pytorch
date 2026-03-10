import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from metagrad import (
    CoordinateMetaGrad,
    FullMetaGrad,
    FullBlockMetagrad,
    SketchedMetaGrad,
    SketchedBlockMetaGrad,
)

OPTIMIZERS = optimizers = [
    (CoordinateMetaGrad, {"sigma": 3.0, "D_inf": 5}),
    (FullMetaGrad, {"sigma": 3.0, "D_inf": 5}),
    (FullBlockMetagrad, {"sigma": 3.0, "D_inf": 5}),
    (
        SketchedMetaGrad,
        {"sigma": 3.0, "D_inf": 5, "sketch_size": 10},
    ),
    (
        SketchedBlockMetaGrad,
        {"sigma": 3.0, "D_inf": 5, "sketch_size": 5},
    ),
]

torch.manual_seed(123)

class LinearModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x @ self.weight


class SimpleNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f1 = nn.Linear(dim, 10)
        self.f2 = nn.Linear(10, 10)
        self.f3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1) # (3, 10, 10) -> (3, 5, 5)
        self.conv2 = nn.Conv2d(3, 3, 2) # (3, 4, 4) -> (3, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)  
        self.fc1 = nn.Linear(3 * 2 * 2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.mark.parametrize("optimizer", OPTIMIZERS, ids=[opt[0] for opt in OPTIMIZERS])
def test_linear_model(optimizer):
    dim = 10
    batch_size = 32
    num_epochs = 10

    model = LinearModel(dim)
    optimizer = optimizer[0](params=model.parameters(), **optimizer[1])

    X = torch.randn(batch_size, dim)
    y = torch.randn(batch_size)

    criterion = nn.MSELoss()

    initial_output = model(X)
    initial_loss = criterion(initial_output, y)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    final_output = model(X)
    final_loss = criterion(final_output, y)

    assert final_loss.item() < initial_loss.item(), (
        "Loss did not  decrease during training"
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS, ids=[opt[0] for opt in OPTIMIZERS])
def test_non_linear_model(optimizer):
    dim = 1
    batch_size = 100
    num_epochs = 100

    model = SimpleNN(dim)
    optimizer = optimizer[0](params=model.parameters(), **optimizer[1])

    X = torch.randn(batch_size, dim)
    y = torch.sin(10 * X) + 0.2 * torch.randn((batch_size, 1))

    criterion = nn.MSELoss()

    initial_output = model(X)
    initial_loss = criterion(initial_output, y)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    final_output = model(X)
    final_loss = criterion(final_output, y)

    assert final_loss.item() < initial_loss.item(), (
        "Loss did not  decrease during training"
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS, ids=[opt[0] for opt in OPTIMIZERS])
def test_cnn_model(optimizer):
    batch_size = 100
    num_epochs = 100

    model = SimpleCNN()
    optimizer = optimizer[0](params=model.parameters(), **optimizer[1])

    # Create image-like data (10x10 images with 1 channel)
    X = torch.randn(batch_size, 1, 10, 10)
    
    # Create non-linear target: sum of sine waves across spatial dimensions
    x_flat = X.view(batch_size, -1)
    y = torch.sum(torch.sin(5 * x_flat), dim=1, keepdim=True) + 0.1 * torch.randn(batch_size, 1)

    criterion = nn.MSELoss()

    initial_output = model(X)
    initial_loss = criterion(initial_output, y)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    final_output = model(X)
    final_loss = criterion(final_output, y)

    assert final_loss.item() < initial_loss.item(), (
        "Loss did not decrease during training"
    )
