# Digital-calibration-of-oysters-with-AI
This is an entreprise project, the purpose of which is to develop a program using deep learning methods for our client so that the quality category of oysters can be automatically identified.

## Introduction
After discussing with our client, we determined the following 7 variables as the criteria for judging the quality of oysters: **Length**, **Width**, **Height**, **SpaceC**, **SpaceD**, **VA** and **VB**.

**SpaceC** is the area between the envelopping circle of the oyster and the outline of the oyster. This variable is observed from the top of the oyster. The smaller the value of this variable, the rounder the oyster.

**SpaceD** has the same meaning as SpaceC. But SpaceD is observed from the side of the oyster.

**VA** is the total area of the protrusions on the surface of the oyster. This variable is observed from the top of the oyster.

**VB** has the same meaning as VA, but VB is observed from the side of the oyster.

According to the values of these 7 variables, the quality of the oyster will be scored 0-10. A score from 8 to 10 is considered as good. A score from 4 to 7 is considered as moderate. A socre from 0 to 3 is considered as bad.

## Dataset
The client provided us with photos of 136 oysters, including top and side views (which means a total of 272 photos), and the score for each oyster. We calculated the values of the above 7 variables for each oyster through image processing, which will be the input of our neural network.

If you are insterested in our pictures of oysters and their calculation results, please consult the [dataset](/) folder.

## Image processing
1. In order to calculate the values of variables more accurately, we have fisrtly transformed the background of the oyster photos to white. We removed the background of oyster pictures and then pasted oysters on another white background.

For more details, please consult the [Rembg](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/image%20processing/Rembg.py/) file.

2. Then, [GetCharacter](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/image%20processing/GetCharacter.py/) program is used to calculate **Length**, **Width**, **Height**, **SpaceC** and **SpaceD**.

3. Next, [GetVariance](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/image%20processing/GetVariance.py/) program is used to calculate **VA** and **VB**.

## Neural network
In view of the small number of variables and the classification task that isn't very complicated, we only used a simple neural network model that is composed of 2 fully connected layers with 50 neurons on each layer.

For more details, please consult the [MLP](https://github.com/Weizhe-JIA/3.Digital-calibration-of-oysters-with-AI/blob/main/network/MLP.py/) program.

## Training

## Application
