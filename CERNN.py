import torch
import torch.nn as nn


class CERNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_expectiles):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, num_expectiles)

    def forward(self, x):

        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        return x


class CERNN_LOSS(nn.Module):

    def __init__(self, expectile_levels):

        super().__init__()
        self.expectile_levels = expectile_levels

    def forward(self, y_pred, y_true, model: CERNN, weight_decay):

        residual = y_true - y_pred
        loss = torch.where(residual >= 0, self.expectile_levels * residual ** 2,
                           (1 - self.expectile_levels) * residual ** 2)

        loss = torch.mean(loss) + weight_decay * torch.sum(model.fc1.weight.data ** 2)

        return loss
