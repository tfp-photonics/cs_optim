import torch
import torch.optim as optim
import torch.nn as nn


class LinearNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nlaf,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlaf = nlaf

        ### create linear model
        self.input = nn.Linear(input_dim, hidden_dim[0])
        self.hidden = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
                for i in range(len(hidden_dim) - 1)
            ]
        )
        self.output = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        x = self.nlaf[0](self.input(x))
        for i in range(len(self.hidden_dim) - 1):
            x = self.nlaf[i + 1](self.hidden[i](x))
        return self.output(x)


class ClassifierNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nlaf,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlaf = nlaf

        ### create linear model
        self.input = nn.Linear(input_dim, hidden_dim[0])
        self.hidden = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
                for i in range(len(hidden_dim) - 1)
            ]
        )
        self.output = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        x = self.nlaf[0](self.input(x))
        for i in range(len(self.hidden_dim) - 1):
            x = self.nlaf[i + 1](self.hidden[i](x))
        return self.output(nn.Softmax()(x))
