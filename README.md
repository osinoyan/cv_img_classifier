## Pytorch Image classification
Code for image classification applying resnet50 using pytorch.

[Github link](https://github.com/osinoyan/cv_img_classifier)

## Environmont
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS

## Outline
[TOC]

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n cv_img_class python=3.6
source activate cv_img_class
pip install -r requirements.txt
```

## Dataset Preparation

### Prepare Images
To download the dataset from the kaggle competition CS_T0828_HW1, run the following command.
Make sure to have *kaggle* installed before using the *kaggle* commands.
```
$ kaggle competitions download -c cs-t0828-2020-hw1
```
After running the command, you will get the zipped file *cs-t0828-2020-hw1.zip*.
Unzip *cs-t0828-2020-hw1.zip* and make sure to place the training and testing image files as the structure below (you must create the directories mannually):
```
    data
    +- train
        +- 000001.jpg
        +- ...
    +- test
        +- 000004.jpg
        +- ...
```
### Convert .csv file to .mat file
To generate a train_list.mat for training, run *make_dataset_list.py* using the following command:
```
$ python make_dataset_list.py
```

## Training

### Train models
To train models, run following commands.
```
$ python main.py -a resnet50 --epochs 100 --pretrained [path_to_dataset]
```

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
resnet50 | 4x TitanXp | 224 | 100 | 70 min

## References
The pytorch implementation based on [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet)