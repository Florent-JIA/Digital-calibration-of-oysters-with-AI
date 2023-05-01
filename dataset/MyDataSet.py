#用于设置x，y，以及划分训练集，验证集，测试集
import torch
import random, csv
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing


class MyDataSet(Dataset):
    """
    MyDataSet implement DataSet of torch, for loading the dataset
    """
    def __init__(self, root, mode):
        """
        :param root: root for dataset
        :param mode: train,val,test
        """
        super(MyDataSet, self).__init__()

        self.mode = mode   # Set the mode of loading the dataset，train
        self.root = root   # root for dataset，一个文件的路径
        label = []
        with open(root, 'r') as f:    # read data from .csv
            reader = csv.reader(f)
            result = list(reader)
            print(result)
            print(result[0])
            del result[0]             # remove the head of form
            random.shuffle(result)
            for i in range(len(result)):
                #print(result[i][0])
                del result[i][0]
                #print(result[i][0])
                #del result[i][0]
                label.append(int(result[i][0]))
                print(label)
                del result[i][0]
                #del result[i][3]

                #print(result)
                #del result[i][3]
        print(result)
        result = np.array(result, dtype=np.float64)
        # result = preprocessing.scale(result).tolist()
        result = preprocessing.StandardScaler().fit_transform(result).tolist()  # 对数据进行预处理

        # result = preprocessing.MinMaxScaler().fit_transform(result).tolist()
        assert len(result) == len(label)
        self.labels = label
        self.datas = result

        if mode == 'train':  # train set 60%
            self.datas = self.datas[:int(0.6 * len(self.datas))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':  # valuation set 20%
            self.datas = self.datas[int(0.6 * len(self.datas)):int(0.8 * len(self.datas))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:  # test set 20%
            self.datas = self.datas[int(0.8 * len(self.datas)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


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
    file_path = "C:/Users/Tardis/Desktop/huit/svg evaluation huitres.csv"
    db = MyDataSet(file_path, 'train')
    x,y = next(iter(db))
    print(x)
    print(y)
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
