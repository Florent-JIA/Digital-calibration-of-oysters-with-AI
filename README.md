# Digital-calibration-of-oysters-with-AI
This is an entreprise project, the purpose of which is to develop a program using deep learning methods for our client so that the quality category of oysters can be automatically identified.

## Introduction
After discussing with our client, we determined the following 7 variables as the criteria for judging the quality of oysters: **Length**, **Width**, **Height**, **SpaceC**, **SpaceD**, **VA** and **VB**.

**SpaceC** is the area between the envelopping circle of the oyster and the outline of the oyster. This variable is observed from the top of the oyster. The smaller the value of this variable, the rounder the oyster.

**SpaceD** has the same meaning as SpaceC. But SpaceD is observed from the side of the oyster.

**VA** is the total area of the protrusions on the surface of the oyster. This variable is observed from the top of the oyster.

**VB** has the same meaning as VA, but VB is observed from the side of the oyster.

<img src="https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/imgs/001d.png" width="500"/><img src="https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/imgs/001c.png" width="500"/>
**(a) top view of an oyster                                                      (b) side view of an oyster**

According to the values of these 7 variables, the quality of the oyster will be scored 0-10. A score from 8 to 10 is considered as good. A score from 4 to 7 is considered as moderate. A socre from 0 to 3 is considered as bad.

The goal of this project is to classify oysters' scores (output) based on these 7 variables (input).

## Dataset
The client provided us with photos of 136 oysters, including top and side views (which means a total of 272 photos), and the score for each oyster. We calculated the values of the above 7 variables for each oyster through image processing, which will be the input of our neural network.

[MyDataSet](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/dataset/MyDataSet.py/) is used to generate the dataset for the training of the network model, including specifying input and output, and dividing training set and test set.

If you are insterested in our pictures of oysters and their calculation results, please consult the [dataset](/) folder.

## Image processing
1. The pictures provided by our client are all in .jpg format. Many functions in the following programs require images in .png format. So, we created the [to_png](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/image_processing/to_png.py/) program to convert all images to .png format.
2. The first class (RemoveBackground) in [MyPreprocessing](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/image_processing/MyPreprocessing.py/) is used to convert the background of images to white.
3. The second class (Calculation) in [MyPreprocessing](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/image_processing/MyPreprocessing.py/) is used to calculate values of Length, Width, Height, SpaceC, SpaceD, VA and VB.
4. In particular, the calculation of VA and VB uses the Fourier transform. The Fourier transform converts an image into a frequency matrix. Those parts whose frequency is low are eliminated. Those parts whose frequency is strong is kept, which are just the parts in relief of the oyster. By an inverse transformation, the new images that keep only the parts in relief of oysters are gotten.
<img src="https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/imgs/FFT2.png" width="300"/><img src="https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/imgs/FFT1.png" width="300"/>
<br>**(a) the parts in relief on the top view of an osyter            (b) the parts in relief on the side view of an osyter**

## Neural network
In view of the small number of variables and the classification task that isn't very complicated, we only used a simple neural network model that is composed of 2 fully connected layers with 50 neurons on each layer.

For more details, please consult the [MLP](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/network/MLP.py/) program.

## Training
run [train_mlp](/) to train the above network model.

## Application
After training, we saved the parameters of the network. In application (test), the photo of an oyster is processed in the same way as in training, which means transforming the background to white and calculating the values of the 7 variables. Then, the neural network classifies the quality score of the oyster.

run [Test_MLP](/) to test and use this neural network.
