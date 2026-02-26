import torch

A = torch.tensor(
    [
        [[1, 1], [2, 2], [3, 3]],
        [[4, 4], [5, 5], [6, 6]],
        [[7, 7], [8, 8], [9, 9]],
    ]
)
mask = torch.tensor([True, False])
print(mask.shape)

B = -1 * torch.tensor(
    [
        [[1, 1], [2, 2], [3, 3]],
        [[4, 4], [5, 5], [6, 6]],
        [[7, 7], [8, 8], [9, 9]],
    ]
)

C = torch.where(mask, B, A)

print(C[:, :, 0])
print(C[:, :, 1])

g = torch.tensor([1, 2, 3, 4])
eta_grid = torch.tensor([-1, -2, -3])

g_g_T = torch.outer(g, g)
print((eta_grid * g_g_T.unsqueeze(-1)).shape)
