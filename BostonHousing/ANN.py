import torch.nn as nn


class ANN(nn.Module):

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x
