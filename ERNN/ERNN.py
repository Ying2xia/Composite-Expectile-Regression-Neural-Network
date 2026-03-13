import torch
import torch.nn as nn


class ERNN(nn.Module):

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


class ExpectileLossWithPenalty(nn.Module):

    def __init__(self, expectile_level):

        super().__init__()
        self.expectile_level = expectile_level

    def forward(self, y_pred, y_true, model: ERNN, weight_decay):

        residual = y_true - y_pred
        loss = torch.where(residual >= 0, self.expectile_level * residual ** 2,
                           (1 - self.expectile_level) * residual ** 2)

        loss = torch.mean(loss) + weight_decay * torch.sum(model.hidden_layer.weight.data ** 2)

        return loss
