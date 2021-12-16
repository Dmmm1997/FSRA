<h1 align="center"> FSRA </h1>

This repository contains the dataset link and the code for our paper [A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization](https://ieeexplore.ieee.org/document/9648201), IEEE Transactions on Circuits and Systems for Video Technology. Thank you for your kindly attention.

## requirement
1. Download the [University-1652](https://github.com/layumi/University1652-Baseline) dataset
2. Configuring the environment
   * First you need to configure the torch and torchision from the [pytorch](https://pytorch.org/) website
   * ```shell
     pip install -r requirement.txt
     ```

## Download pre-training weights
You can download the pre-training weight from the following link and put them in the **pretrain_model** folder

1. [Baidu Driver](https://pan.baidu.com/s/1ci2rNHb4ARKCwyVemT_JNg)  code: 52n2

2. [Google Driver](https://drive.google.com/file/d/1gEfXVVdL827J5SJ4KGOHhdWoeDZ7UjTd/view?usp=sharing)

## Train and Test
We provide scripts to complete FSRA training and testing
* Change the **data_dir** and **test_dir** paths in **train_test_local.sh** and then run:
```shell
bash train_test_local.sh
```
