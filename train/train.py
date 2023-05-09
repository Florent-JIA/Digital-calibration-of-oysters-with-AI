import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataset.CreateDataset import MyDataSet
from network.network import MLP
import sklearn.metrics as m
import warnings

warnings.filterwarnings("ignore")

class Options:
    def __init__(self):
        self.root = r'dataset.csv'
        self.batchsz = 2
        self.lr = 0.01
        self.epoches = 10
        self.TrainSet = 0.6
        self.ValSet = 0.2
        self.TestSet = 0.2


torch.manual_seed(1234)

opt = Options()

train_db = MyDataSet(opt, mode='train')
val_db = MyDataSet(opt, mode='val')
test_db = MyDataSet(opt, mode='test')

train_loader = DataLoader(train_db, batch_size=opt.batchsz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=opt.batchsz)
test_loader = DataLoader(test_db, batch_size=opt.batchsz)


def evaluate(model, loader):
    """
    :param model: net model
    :param loader: data set
    :return: ACC
    """
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def _test_evaluate(model, loader):
    """
       :param model: net model
       :param loader: data set
       :return: ACC
    """
    y_true = []
    predict = []
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            result = logits.argmax(dim=1)
            for i in y.numpy():
                y_true.append(i)
            for j in result.numpy():
                predict.append(j)

    print(classification_report(y_true, predict))

    print("confusion matrix")
    print(confusion_matrix(y_true, predict))
    print("f1-score:{}.".format(m.f1_score(y_true, predict, average='weighted')))

    print(y_true)
    print(predict)
    return accuracy_score(y_true, predict)


if __name__ == '__main__':

    model = MLP(7, 5)  # initial net model
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)  # Adam optimizer
    criterion = nn.CrossEntropyLoss()  # loss function

    best_epoch, best_acc = 0, 0

    for epoch in range(opt.epoches):
        for step, (x, y) in enumerate(train_loader):
            # x:[b,16] ,y[b]
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()  # mettre en zero les gradients
            loss.backward()  # transmission inverse
            optimizer.step()  # réduction des gradients

        if epoch % 5 == 0:

            val_acc = evaluate(model, val_loader)
            train_acc = evaluate(model,train_loader)
            print("epoch:[{}/{}]. val_acc:{}.".format(epoch, opt.epoches, val_acc))

            print("train_acc", train_acc)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')

    print('best acc:{}. best epoch:{}.'.format(best_acc, best_epoch))
    model.load_state_dict(torch.load('best.mdl'))
    print("loaded from ckpt!")

    test_acc = _test_evaluate(model, test_loader)
    print("test_acc:{}".format(test_acc))
