from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing


class CaliforniaHousing(Dataset):

    def __init__(self, K=1, train=True):

        data = fetch_california_housing()
        self.data = data['data'].astype('float32')
        self.target = data['target'].astype('float32')
        self.target = self.target.repeat(K).reshape(len(self.target), -1)

        if train:
            self.data = self.data[: 12384]
            self.target = self.target[: 12384]
        else:
            self.data = self.data[12384:]
            self.target = self.target[12384:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
