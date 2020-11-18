# Seismic-Facies-classification

This is a pytorch version of seismic facies classification 
---
## Dataset
The training and validation data can be downloaded from OLIVES group (https://github.com/olivesgatech/facies_classification_benchmark). 
![data](https://github.com/yxycug/Seismic-Facies-classification/blob/master/result/data.png)

---
## Abstract
Intelligent seismic facies classification based on deep learning methods can greatly reduce manual operations. Conventional deep learning methods used for seismic facies recognition, the network model can only extract feature map under a single receptive field, and it is difficult to obtain the global spatial distribution information of seismic section. And the prediction of minority seismic facies boundary is inaccurate. It is also hard for multi-class segmentation models to uncertainty estimation. Given these issues, we implement our facies classification network by simplified U-Net with pyramid pooling module which empirically proved to be an effective global contextual prior. And we used an objective function combining cross-entropy and Dice loss which can improve the boundary characterization of minority seismic facies in unbalanced data. We present Prediction Entropy for estimating the uncertainty of classification results. The application of our scheme on F3 dataset demonstrates its improvements and we observe Prediction Entropy can evaluates the uncertainty of the prediction results well.

---
## Model

![model](https://github.com/yxycug/Seismic-Facies-classification/blob/master/result/unet-ppm_model.png)



---
## Prediction

![patch](https://github.com/yxycug/Seismic-Facies-classification/blob/master/result/prediction.png)
