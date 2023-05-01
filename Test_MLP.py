# train_mlp.py
import torch
from torch.utils.data import DataLoader
from TestDataSet import TestDataSet
from MLP import MLP
from Rembg import RemoveBackground
from GetCharacter import GetCharacter
class TestMlp:
    """
    TestMlp for processing the images and getting the predict of the quality of the huit
    """
    Result = [] #The final predict of the quality of the huit
    def __init__(self, inputroot1,inputroot2,outputClass,realSize1,realSize2):
        """
        :param inputroot1: root for the input of the original image A(Vertical view)
        :param inputroot2: root for the input of the original image B(Side view)
        :param outputClass: The types of huit you want to classify into, choose from 3,5,10
        :param realSize1: The real size of the area covered by the image A (Vertical view)
        :param realSize2: The real size of the area covered by the image B (Side view)
        """
        self.inputroot1 = inputroot1
        self.inputroot2 = inputroot2
        self.outputClass = outputClass
        self.realSize1 = realSize1
        self.realSize2 = realSize2

        print("inputroot1 = "+ self.inputroot1 +", inputroot2 = "+ self.inputroot2 + ", outputClass = " + str(self.outputClass))

        #Remove the backgrounds of the images and change them into white
        xx = RemoveBackground(self.inputroot1, self.inputroot2)
        xx.removeBackground()
        print("Background removed")
        xx.backgroundToWhite()
        print("Background changed to white")

        #Get the Hight, Width, length, SpaceA, SpaceB, VA and VB of the huit from the images
        GetCharacter(self.realSize1,self.realSize2)


        batchsz = 1
        torch.manual_seed(1234)
        file_path = "out.csv"

        # load data
        test_db = TestDataSet(file_path, mode='test')
        test_loader = DataLoader(test_db, batch_size=batchsz)


        # Calculate the predict
        def test_evaluate(model, loader):
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





            #print(predict)
            self.Result = predict

        #acoording to the types of huit you want to classify them into, we load different model.
        if (self.outputClass == 5):
            model = MLP(7, 5)  # initial net model
            model.load_state_dict(torch.load('best_5.mdl'))
        elif  (self.outputClass == 3):
            model = MLP(7, 3)  # initial net model
            model.load_state_dict(torch.load('best_3.mdl'))
        elif  (self.outputClass == 10):
            model = MLP(7, 10)  # initial net model
            model.load_state_dict(torch.load('best_10.mdl'))
        else:
            print("Impossible to classify the huits into "+ str (self.outputClass)+" classed, please change it into 3, 5 or 10")

        #print("loaded from ckpt!")

        test_evaluate(model, test_loader)
        print("Result = ")
        #print("test_acc:{}".format(test_acc))






def main():
    xx = TestMlp("001d.jpg","001c.jpg",3,1,1)
    print(xx.Result)


if __name__ == '__main__':
    main()
