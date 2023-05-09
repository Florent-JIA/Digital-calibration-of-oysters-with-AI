import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from network import MLP
from MyPreprocessing import RemoveBackground
from MyPreprocessing import Calculation


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]



def Application(inputroot1, inputroot2, realSize1=1, realSize2=1):
    xx = RemoveBackground(inputroot1, inputroot2)
    p1, p2 = xx.backgroundToWhite()

    Cal = Calculation(p2, p1, realSize1, realSize2)
    L, W, H, SpaceC, SpaceD = Cal.GetCharacter()
    VA, VB = Cal.GetVariance()

    input = torch.tensor([L, W, H, SpaceC, SpaceD, VA, VB]).float()
    label = torch.tensor([2])

    inputs = [input]
    labels = [label]

    testset = CustomDataset(inputs, labels)

    test_loader = DataLoader(testset, batch_size=1)

    model = MLP(7, 3)
    model.load_state_dict(torch.load('best.mdl'))

    predict = []
    for x, y in test_loader:
        with torch.no_grad():
            logits = model(x)
            result = logits.argmax(dim=1)
            for j in result.numpy():
                predict.append(j)

    print(f"Result is {predict}")


input1 = r"001c.png"
input2 = r"001d.png"

Application(input1, input2)



