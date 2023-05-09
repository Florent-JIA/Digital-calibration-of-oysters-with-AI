import os
import pandas as pd
import glob
from image_processing import MyPreprocessing
import torch
import random, csv
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing

def CreateDataset(opt):
    DatasetDir = opt.DatasetDir
    Label = opt.Label
    LabelTable = pd.read_excel(Label)

    df = pd.read_excel(Label, header=0)
    df_column1 = df.iloc[:, 0]

    df = pd.DataFrame(columns=["ID", "Label", "L", "W", "H", "SpaceC", "SpaceD", "VA", "VB"])

    OystersLackingPics = []
    RowInDataset = 1
    for folder_name in df_column1:
        folder_path = os.path.join(DatasetDir, str(folder_name))

        img_paths = glob.glob(os.path.join(folder_path, '*.png'))
        if len(img_paths) == 2:
            inputroot1 = img_paths[0]
            inputroot2 = img_paths[1]

            RemoveBackground = MyPreprocessing.RemoveBackground(inputroot1, inputroot2)
            p1, p2 = RemoveBackground.backgroundToWhite()

            Calculation = MyPreprocessing.Calculation(p2, p1, 1, 1)

            L, W, H, SpaceC, SpaceD = Calculation.GetCharacter()
            VA, VB = Calculation.GetVariance()

            df.at[RowInDataset, "ID"] = folder_name
            ValueLabel = LabelTable.iloc[RowInDataset-1, 1]
            df.at[RowInDataset, "Label"] = ValueLabel
            df.at[RowInDataset, "L"] = L
            df.at[RowInDataset, "W"] = W
            df.at[RowInDataset, "H"] = H
            df.at[RowInDataset, "SpaceC"] = SpaceC
            df.at[RowInDataset, "SpaceD"] = SpaceD
            df.at[RowInDataset, "VA"] = VA
            df.at[RowInDataset, "VB"] = VB

            RowInDataset += 1
            break
        else:
            OystersLackingPics.append(folder_name)

    df.to_csv(os.path.join(opt.SaveDir, 'dataset.csv'), index=False)


class MyDataSet(Dataset):
    """
    MyDataSet implement DataSet of torch, for loading the dataset
    """
    def __init__(self, opt, mode):
        """
        :param root: root for dataset
        :param mode: train,val,test
        """
        super(MyDataSet, self).__init__()

        self.mode = mode  # Set the mode of loading the dataset，train
        self.root = opt.root
        label = []
        with open(self.root, 'r') as f:    # read data from .csv
            reader = csv.reader(f)
            result = list(reader)
            del result[0]  # remove the head of form
            random.shuffle(result)
            for i in range(len(result)):
                del result[i][0]
                label.append(int(result[i][0]))
                del result[i][0]


        result = np.array(result, dtype=np.float64)
        result = preprocessing.StandardScaler().fit_transform(result).tolist()

        assert len(result) == len(label)
        self.labels = label
        self.datas = result

        if self.mode == 'train':  # train set 60%
            self.datas = self.datas[:int(opt.TrainSet * len(self.datas))]
            self.labels = self.labels[:int(opt.TrainSet * len(self.labels))]
        elif self.mode == 'val':  # valuation set 20%
            self.datas = self.datas[int(opt.TrainSet * len(self.datas)):int((1-opt.ValSet) * len(self.datas))]
            self.labels = self.labels[int(opt.TrainSet * len(self.labels)):int((1-opt.ValSet) * len(self.labels))]
        else:  # test set 20%
            self.datas = self.datas[int((1-opt.TestSet) * len(self.datas)):]
            self.labels = self.labels[int((1-opt.TestSet) * len(self.labels)):]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # idx~[0~len(data)]
        data, label = self.datas[idx], self.labels[idx]
        data = torch.tensor(data)
        label = torch.tensor(label)
        return data, label


if __name__ == '__main__':

    class Options():
        def __init__(self):
            self.root = r'C:\fise a3\stage A3\25.Segula_2\pyhuit\dataset\photos1\dataset.csv'
            self.TrainSet = 0.6
            self.ValSet = 0.2
            self.TestSet = 0.2

    opt = Options()

    train_db = MyDataSet(opt, mode='train')
    val_db = MyDataSet(opt, mode='val')
    test_db = MyDataSet(opt, mode='test')

    for data, label in train_db:
        print(data, label)