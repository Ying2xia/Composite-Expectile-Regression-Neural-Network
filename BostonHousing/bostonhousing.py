import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BostonHousing(Dataset):

    def __init__(self, K=1, train=True):

        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        self.data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]).astype('float32')
        self.target = raw_df.values[1::2, 2].astype('float32')
        self.target = self.target.repeat(K).reshape(len(self.target), -1)

        if train:
            self.data = self.data[: 400]
            self.target = self.target[: 400]
        else:
            self.data = self.data[400:]
            self.target = self.target[400:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
