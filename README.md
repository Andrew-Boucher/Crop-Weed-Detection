# <center>Crop and Weed Object Detection</center>
![Robot%20Precision%20Agriculture.jpeg](attachment:Robot%20Precision%20Agriculture.jpeg)

## Overview

Weeds have always been a problem in edible crop agriculture. They compete with crops for vital resources and can reduce their overall production. One of the most common practices to deal with weeds is to “broadcast” spray foliar herbicide over an entire field. However, herbicide usage in crop production can have many unintended negative effects not only on the environment but the humans who end up ingesting the crops as well. Currently, there are several companies who are working on solutions to this problem. One approach involves using robots who rely on machine vision to selectively spray herbicide or use some other elimination technique only on the areas that contain the weeds. In order to do this the robot must be able to distinguish the difference between the weed and the edible crop. Additionally, the robot must also be able to locate (draw bounding boxes around) both the weed and crop in order to properly preserve the crop and eliminate the weed. Below are the steps necessary for developing an object detection model which would the driving mechanism behind this machine vision task. Ultimately, this would allow farmers to reduce the use of herbicides, potentially lowering production costs as well as improving the health of humans and the planet.

## Data Understanding

The data used in this project was sourced from a [Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes) dataset. The dataset contained 546 origial colored (RGB) 512x512 images of sesame crops and weeds. The original images were augmented to produce 1300 unique images total in the dataset. Each image had an associated .txt file annotation in YOLO format which contained the ground truth class labels and bounding box locations. The YOLO annotation format is as follows: 

(Class Label, Bounding Box "X" Center Coordinate, Bounding Box "Y" Center Coordinate, Bounding Box Width, Bounding Box Height)

## Data Preperation

Once the data was loaded, the labels were abstracted from the .txt file and put into a dataframe where it was reformated from YOLO into Pascal VOC format. Then the images dimensions were reduced and the pixel intensities were scaled. Then the data was split into trainging, validation and test sets.

## Modeling

For the modeling process we utlized transfer learning techniques by using the structure and weights from seven different keras models which were pretrained on the imagenet dataset to create the 'backbone' of our model. These models included Xception, VGG16, ResNet152V2, MobileNetV2, DenseNet201 and NASNetLarge. We also created one custom simple model "backbone" which was trained on the crop and weed dataset. The "backbones" were connected to a pair of different model "heads", one for predicting class lables and one for predicting bounding box labels. We then used the training loss information and the COCO mean average precision (mAP) metric to iterate through several different versions of the model heads for each backbone. See our final notebook to see details about the different models and their results during training. 

## Results

The current standard for object detection model metrics is the COCO Mean Average Precision score so we will be using that as our main driving metric for our models' evaluation. However, we feel it is important to also consider the inference time as secondary metric due to potential processing constraints in the final model's deployment use case.

Below is a graph showing COCO Mean Average Precisions scores on the validation set for all the models:
![COCO mAP Val](../Plots/COCO mAP for Crop & Weed Detection Models)

Looks like the model with the VGG16 backbone and the 6th version of the model heads had the highest COCO Mean Average Precision score. It's COCO mAP score on the validation set was 55.3%.

Next we will look at our secondary metric the average inference time the model takes to make a predition on an image. Below is a graph showing the Average Inference Time (in seconds) for the images in the validation set for all the models:
![Inference Time Val]("../Plots/Average Inference Time Per Image (CPU) for Crop & Weed Detection Models")

Looks like the model with the custom backbone and the 7th version of the model heads had the lowest average inference time. It's average inference time on the validation set was 0.039 seconds.

## Conclusions

Now lets evaluate both those models on the test set. However for comparitive purposes, since custom_model_v7 and custom model_v1 had nearly identical average inference times, only a difference of 0.001 seconds, lets compare the best COCO mAP score model to the custom model with the best COCO mAP score as well.

Below is a graph showing the COCO Mean Average Precision scores the test set for the final models:

![COCO mAP Test]("../Plots/COCO mAP for Crop & Weed Detection Final Models")

As you can see the model with the VGG16 backbone and 6th version of the model heads had the highest COCO mAP score on the test set. Its COCO mAP score on the test set was 56.7% which is acutally higher than its score on the validation set (55.3%). It is also interesting to note that the model with the custom backbone and 1st version of the model heads, which was much faster than its vgg16 counterpart, 0.037 seconds compared to 0.164 seconds, respecitvely, only had a 1.4% reduction in COCO mAP score on the test set at 55.3% when compared to VGG16 backbone model (56.7%).

## Repository Structure

├── Data                               <- Source datasets used in this analysis
├── Plots                              <- All the plots created during training and final evaluations
├── Models                             <- Trained models' weights and parameters saved in HDF5 format
├── Evaluation                         <- Models' metric results saved as csv
├── .gitignore                         <- .gitignore file to prevent git from crawling unwanted and irrelevant files
├── Final_Notebook.ipynb               <- Jupyter Notebook of our methods for the modeling process, results and conclusion
├── README.md                          <- This README
└── Workbook.ipynb                     <- Jupyter Notebook used as our "scratch" notebook