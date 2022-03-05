# Medical-data-classification

The purpose of this thesis is medical data classification using Python deep learning
networks. Two types of neural networks were used in the study: convolutional neural
network and recurrent neural network long short-term memory type. Both neural ne-
tworks were implemented using the PyTorch library. The paper presents the results
of experiments aimed at selecting appropriate network parameters in order to obtain
the best classification accuracy. There were used four data sets: Breast Cancer, Breast
Cancer Wisconsin (Diagnostic), Parkinsons, Heart Disease.

Datasets are provided from: https://archive.ics.uci.edu/ml/index.php

Other implememented functions:
- Adaptive Learning rate (https://uk.mathworks.com/help/deeplearning/ref/traingda.html)
- Cross-validation

CNN 1D Networks:

![cnn](https://user-images.githubusercontent.com/33430525/156878188-e484f686-92ff-42ec-905e-76afd2404655.png)


Paramaters test result for CNN:

Heart Disease Dataset:

![cnn_heartdisease](https://user-images.githubusercontent.com/33430525/156878208-eff08f28-3839-48c6-9a5e-c8eead6fae20.png)

Breast Cancer Dataset:

![cnn_breastcancer](https://user-images.githubusercontent.com/33430525/156878237-3852f49f-3172-45a2-abe5-8cb372dd7a92.png)

Parkinsons Dataset:

![cnn_parkinson](https://user-images.githubusercontent.com/33430525/156878263-a5ebaee3-3d9a-45e2-a71a-5f99388cc6a2.png)

Breast Cancer Diagnostic Dataset:

![cnn_breastcancer_diagnostic](https://user-images.githubusercontent.com/33430525/156878286-2e4e27b5-2c61-4b39-8351-6f9573261b78.png)
