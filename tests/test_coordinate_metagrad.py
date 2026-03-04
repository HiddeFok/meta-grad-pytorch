import torch
import torch.nn as nn

from metagrad import CoordinateMetaGrad


class LinearModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x @ self.weight


def test_coordinate_metagrad_linear_model():
    dim = 10
    batch_size = 32
    num_epochs = 10

    model = LinearModel(dim)
    optimizer = CoordinateMetaGrad(params=model.parameters(), sigma=1, D_inf=10)

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
