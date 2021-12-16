<h1 align="center"> FSRA </h1>

This repository contains the dataset link and the code for our paper [A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization](https://ieeexplore.ieee.org/document/9648201), IEEE Transactions on Circuits and Systems for Video Technology. Thank you for your kindly attention.

## requirement
1. Download the [University-1652](https://github.com/layumi/University1652-Baseline) dataset
2. Configuring the environment
   * First you need to configure the torch and torchision from the [pytorch](https://pytorch.org/) website
   * ```shell
     pip install -r requirement.txt
     ```


## Train and Test
We provide scripts to complete FSRA training and testing
```shell
bash train_test_local.sh
```
