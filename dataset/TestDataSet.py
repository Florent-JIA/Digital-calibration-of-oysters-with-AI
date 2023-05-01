#TestDataSet.py
import torch
import random, csv
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing


class TestDataSet(Dataset):
    """
    TestDataSet implement DataSet of torch, for loading the dataset
    """
    def __init__(self, root, mode):
        """
        :param root: root for dataset
        :param mode: train,val,test
        """
        super(TestDataSet, self).__init__()

        self.mode = mode   # Set the mode of loading the dataset
        self.root = root   # root for dataset
        label = []
        with open(root, 'r') as f:    # read data from .csv
            reader = csv.reader(f)
            result = list(reader)
            del result[0]             # remove the head of form
            random.shuffle(result)
            for i in range(len(result)):
                del result[i][0]
                label.append(int(result[i][0]))
                del result[i][0]

        result = np.array(result, dtype=np.float64)
        # result = preprocessing.scale(result).tolist()
        result = preprocessing.StandardScaler().fit_transform(result).tolist()

        # result = preprocessing.MinMaxScaler().fit_transform(result).tolist()
        assert len(result) == len(label)
        self.labels = label
        self.datas = result



    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # idx~[0~len(data)]
        data, label = self.datas[idx], self.labels[idx]
        data = torch.tensor(data)
        label = torch.tensor(label)
        return data, label
        #print(data)
        #print(label)


def main():
    file_path = "out.csv"
    db = TestDataSet(file_path, 'train')
    x,y = next(iter(db))
    print(x)
    print(y)
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
