from torch import nn

import torch as th


class MLP(nn.Module):
    """
    Default MLP used in SB3 (without squashing, see MLPPolicy).
    """

    def __init__(self, input_size, output_size, layers=2, hidden_size=64, **kwargs):
        super().__init__()
        modules = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * layers),
            nn.Linear(hidden_size, output_size),
        ]

        self.linear = nn.Sequential(*modules)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)


class CNN(nn.Module):
    """
    Modeled after LeNet-5.
    """

    def __init__(self, input_dim, output_size, **kwargs):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim[0], 6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.rand(tuple((1, *input_dim))).float()).shape[-1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, output_size),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(x))
