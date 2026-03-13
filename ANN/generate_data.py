import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import norm, t, chi2, uniform


class UnivariateData(Dataset):

    def __init__(self, n_samples, expectile_level, distribution='normal'):

        self.n_samples = n_samples

        self.x = uniform.rvs(loc=-4, scale=8, size=(n_samples, 1)).astype(np.float32)
        self.y = (1 - self.x + 2 * self.x ** 2) * np.exp(-self.x ** 2 / 2) + self.x ** 2 / 8
        mid = (1 + 0.2 * self.x) / 5

        if distribution == 'norm':
            self.y += mid * norm.ppf(expectile_level, loc=0, scale=1)
        elif distribution == 't':
            self.y += mid * t.ppf(expectile_level, df=3)
        elif distribution == 'chi2':
            self.y += mid * chi2.ppf(expectile_level, df=3)
        else:
            raise ValueError('Unknown distribution')

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

    def __len__(self):
        return self.n_samples


class TrivariateData(Dataset):

    def __init__(self, n_samples, expectile_level, distribution='normal'):

        self.n_samples = n_samples

        self.x1 = np.random.uniform(-1, 2, size=(n_samples, 1)).astype(np.float32)
        self.x2 = np.random.uniform(-1, 2, size=(n_samples, 1)).astype(np.float32)
        self.x3 = np.random.uniform(-1, 2, size=(n_samples, 1)).astype(np.float32)
        self.x = np.concatenate([self.x1, self.x2, self.x3], axis=1)
        self.y = self.x1 ** 2 + np.sin(2 * self.x2) + np.exp(self.x3 ** 2)

        if distribution == 'norm':
            self.y += norm.ppf(expectile_level, loc=0, scale=1)
        elif distribution == 't':
            self.y += t.ppf(expectile_level, df=3)
        elif distribution == 'chi2':
            self.y += chi2.ppf(expectile_level, df=3)
        else:
            raise ValueError('Unknown distribution')

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

    def __len__(self):
        return self.n_samples


class QuadrivariateData(Dataset):

    def __init__(self, n_samples, expectile_level, distribution='normal'):

        self.n_samples = n_samples

        self.x1 = np.random.uniform(0, 1, size=(n_samples, 1)).astype(np.float32)
        self.x2 = np.random.uniform(-1, 1, size=(n_samples, 1)).astype(np.float32)
        self.x3 = np.random.uniform(-3, 3, size=(n_samples, 1)).astype(np.float32)
        self.x4 = np.random.uniform(-3, 3, size=(n_samples, 1)).astype(np.float32)
        self.x = np.concatenate([self.x1, self.x2, self.x3, self.x4], axis=1)
        self.y = np.sin(2 * self.x1) + 2 * np.exp(-16 * self.x2 ** 2) + self.x3 * self.x4

        if distribution == 'norm':
            self.y += norm.ppf(expectile_level, loc=0, scale=1)
        elif distribution == 't':
            self.y += t.ppf(expectile_level, df=3)
        elif distribution == 'chi2':
            self.y += chi2.ppf(expectile_level, df=3)
        else:
            raise ValueError('Unknown distribution')

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

    def __len__(self):
        return self.n_samples
