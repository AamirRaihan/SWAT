# [Sparse Weight Activation Training <img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>](https://papers.nips.cc/paper/2020/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf)

[Md Aamir Raihan](https://www.linkedin.com/in/aamir-raihan-45368052/?originalSubdomain=ca), [Tor Aamodt](https://www.ece.ubc.ca/~aamodt/)

This repository contains code for the CNN experiments presented in the NeurIPS 2020 [paper](https://papers.nips.cc/paper/2020/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf) along with some additional functionalities. 

Table of Contents
=================

   * [<a href="https://papers.nips.cc/paper/2020/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf" rel="nofollow">Sparse Weight Activation Training <img src="h
ttps://camo.githubusercontent.com/b1dff6a6513fce2ebb171af6d4c6e446b6552dadfd6f15f9f71d0d8b1c8b7e26/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f7
7696b6970656469612f656e2f7468756d622f302f30382f4c6f676f5f666f725f436f6e666572656e63655f6f6e5f4e657572616c5f496e666f726d6174696f6e5f50726f63657373696e675f53797
374656d732e7376672f3132303070782d4c6f676f5f666f725f436f6e666572656e63655f6f6e5f4e657572616c5f496e666f726d6174696f6e5f50726f63657373696e675f53797374656d732e737
6672e706e67" width="200" data-canonical-src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems
.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" style="max-width:100%;"></a>](#sparse-weight-activation-training-)
      * [Experiment Setup](#experiment-setup)
         * [Manual Setup](#manual-setup)
         * [Docker Setup](#docker-setup)
      * [Basic Usage](#basic-usage)
         * [Unstructured SWAT](#unstructured-swat)
            * [CIFAR10](#cifar10)
            * [CIFAR100](#cifar100)
            * [IMAGENET](#imagenet)
         * [Structured SWAT](#structured-swat)
            * [CIFAR10](#cifar10-1)
            * [CIFAR100](#cifar100-1)
            * [IMAGEMENT](#imagement)
      * [Pretrained Model](#pretrained-model)
      * [Training/Inference FLOP count](#traininginference-flop-count)
      * [Citation](#citation)

## Experiment Setup

All the experiments can be recreated either by running the docker image or by manually installing the cuda/cudnn with proper dependencies.

### Manual Setup

```bash
# GCC Version  5.5.0 20171010

# Install CUDA-10.0.130 
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
chmod +x cuda_10.0.130_410.48_linux
./cuda_10.0.130_410.48_linux

# Install cuDNN-/cudnn-7.6.4.38 
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.4.38/Production/10.0_20190923/cudnn-10.0-linux-x64-v7.6.4.38.tgz
# Extract cudnn-10.0-linux-x64-v7.6.4.38.tgz
cp libcudnn.so  cuda-10/lib64/
cp cudnn.h   cuda-10/include/

# Install the anaconda enabled with Python 3.7.4
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
chmod +x Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh

#Install Pytorch
# Don't use any other version since the pytorch C++ interface for cuDNN wrapper has been changed and therefore the other version is not compatible with this code.
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# Clone the SWAT Repository
git clone https://github.com/AamirRaihan/SWAT.git

# Install the cuDNN C++ wrapper for custom convolution layer.
cd SWAT/SWAT-code/mypkg
python setup.py install
```

### Docker Setup

You can also pull a pre-built docker image from Docker Hub and run with docker v19.03+

```bash
sudo docker run --gpus all --ipc=host  -it  --rm -v /PATH/TO/IMAGENET/DATASET:/workspace/datasets/ swaticml/custom-swat-pytorch:v1
```

More Information on installing docker is present [here](). 

## Basic Usage

### Unstructured SWAT

#### CIFAR10

|   Network    |  Method  | Weight Sparsity (%) | Activation Sparsity (%) | Top-1 Accuracy (%) | Training FLOP ⬇ % |
| :----------: | :------: | :-----------------: | :---------------------: | :----------------: | :---------------: |
|    VGG-16    |  SWAT-U  |        90.0         |          90.0           |     91.95±0.06     |       89.7        |
|    VGG-16    | SWAT-ERK |        95.0         |          82.0           |     92.50±0.07     |       89.5        |
|    VGG-16    |  SWAT-M  |        95.0         |          65.0           |     93.41±0.05     |       64.0        |
|   WRN-16-8   |  SWAT-U  |        90.0         |          90.0           |     95.13±0.11     |       90.0        |
|   WRN-16-8   | SWAT-ERK |        95.0         |          84.0           |     95.00±0.12     |       91.4        |
|   WRN-16-8   |  SWAT-M  |        95.0         |          78.0           |     94.97±0.04     |       86.3        |
| DenseNet-121 |  SWAT-U  |        90.0         |          90.0           |     94.48±0.06     |       89.8        |
| DenseNet-121 | SWAT-ERK |        90.0         |          88.0           |     94.14±0.11     |       89.7        |
| DenseNet-121 |  SWAT-M  |        90.0         |          86.0           |     94.29±0.11     |       84.2        |

#### CIFAR100

|   Network    |  Method  | Weight Sparsity(%) | Activation Sparsity(%) | Top-1 Accuracy(%) | Training FLOP ⬇ % |
| :----------: | :------: | :----------------: | :--------------------: | :---------------: | :---------------: |
|    VGG-16    |  SWAT-U  |        90.0        |          90.0          |    91.95±0.08     |                   |
|    VGG-16    | SWAT-ERK |        90.0        |          69.6          |    92.50±0.11     |                   |
|    VGG-16    |  SWAT-M  |        90.0        |          59.9          |    93.41±0.23     |                   |
|   WRN-16-8   |  SWAT-U  |        90.0        |          90.0          |    95.13±0.13     |                   |
|   WRN-16-8   | SWAT-ERK |        90.0        |          77.6          |    95.00±0.07     |                   |
|   WRN-16-8   |  SWAT-M  |        90.0        |          73.3          |    94.97±0.11     |                   |
| DenseNet-121 |  SWAT-U  |        90.0        |          90.0          |    94.48±0.06     |                   |
| DenseNet-121 | SWAT-ERK |        90.0        |          90.0          |    94.14±0.03     |                   |
| DenseNet-121 |  SWAT-M  |        90.0        |          84.2          |    94.29±0.13     |                   |

#### IMAGENET

|  Network  |  Method  | Weight Sparsity (%) | Activation Sparsity (%) | Top-1 Accuracy (%) | Training FLOP ⬇ % |
| :-------: | :------: | :-----------------: | :---------------------: | :----------------: | :---------------: |
| ResNet-50 |  SWAT-U  |        80.0         |          80.0           |     75.2±0.06      |       76.1        |
| ResNet-50 |  SWAT-U  |        90.0         |          90.0           |     72.1±0.03      |       85.6        |
| ResNet-50 | SWAT-ERK |        80.0         |          52.0           |     76.0±0.16      |       60.0        |
| ResNet-50 | SWAT-ERK |        90.0         |          64.0           |     73.8±0.23      |       79.0        |
| ResNet-50 |  SWAT-M  |        80.0         |          49.0           |     74.6±0.10      |       45.9        |
| ResNet-50 |  SWAT-M  |        90.0         |          57.0           |     74.0±0.18      |       65.4        |
| WRN-50-2  |  SWAT-U  |        80.0         |          80.0           |     76.4±0.11      |       78.6        |
| WRN-50-2  |  SWAT-U  |        90.0         |          90.0           |     74.7±0.27      |       88.4        |

### Structured SWAT

#### CIFAR10

|   Network    | Method | Weight Sparsity (%) | Activation Sparsity (%) | Channel Pruned (%) | Top-1 Accuracy (%) | Training FLOP ⬇ % |
| :----------: | :----: | :-----------------: | :---------------------: | :----------------: | :----------------: | :---------------: |
|  ResNet-18   | SWAT-U |        50.0         |          50.0           |        50.0        |     94.73±0.06     |       49.9        |
|  ResNet-18   | SWAT-U |        60.0         |          60.0           |        60.0        |     94.68±0.03     |       59.8        |
|  ResNet-18   | SWAT-U |        70.0         |          70.0           |        70.0        |     94.65±0.19     |       69.8        |
| DenseNet-121 | SWAT-U |        50.0         |          50.0           |        50.0        |     95.04±0.26     |       49.9        |
| DenseNet-121 | SWAT-U |        60.0         |          60.0           |        60.0        |     94.82±0.11     |       59.9        |
| DenseNet-121 | SWAT-U |        70.0         |          70.0           |        70.0        |     94.81±0.20     |       69.9        |

#### CIFAR100

|   Network    | Method | Weight Sparsity (%) | Activation Sparsity (%) | Channel Pruned (%) | Top-1 Accuracy (%) |
| :----------: | :----: | :-----------------: | :---------------------: | :----------------: | :----------------: |
|  ResNet-18   | SWAT-U |        50.0         |          50.0           |        50.0        |     76.4±0.05      |
|  ResNet-18   | SWAT-U |        60.0         |          60.0           |        60.0        |     76.2±0.11      |
|  ResNet-18   | SWAT-U |        70.0         |          70.0           |        70.0        |     75.6±0.09      |
| DenseNet-121 | SWAT-U |        50.0         |          50.0           |        50.0        |     78.7±0.03      |
| DenseNet-121 | SWAT-U |        60.0         |          60.0           |        60.0        |     78.5±0.08      |
| DenseNet-121 | SWAT-U |        70.0         |          70.0           |        70.0        |     78.1±0.12      |

#### IMAGEMENT

|  Network  | Method | Weight Sparsity (%) | Activation Sparsity (%) | Channel Pruned (%) | Top-1 Accuracy (%) | Training FLOP ⬇ % |
| :-------: | :----: | :-----------------: | :---------------------: | :----------------: | :----------------: | :---------------: |
| ResNet-50 | SWAT-U |        50.0         |          50.0           |        50.0        |     76.51±0.30     |       47.6        |
| ResNet-50 | SWAT-U |        60.0         |          60.0           |        60.0        |     76.35±0.06     |       57.1        |
| ResNet-50 | SWAT-U |        70.0         |          70.0           |        70.0        |     75.67±0.06     |       66.6        |
| WRN-50-2  | SWAT-U |        50.0         |          50.0           |        50.0        |     78.08±0.20     |       49.1        |
| WRN-50-2  | SWAT-U |        60.0         |          60.0           |        60.0        |     77.55±0.07     |       58.9        |
| WRN-50-2  | SWAT-U |        70.0         |          70.0           |        70.0        |     77.19±0.11     |       68.7        |

## Pretrained Model



## Training/Inference FLOP count



## Citation

If you find this project useful in your research, please consider citing:

```
​```
@inproceedings{RaihanNeurips2020
  author    = {Raihan, Md Aamir and Aamodt, Tor M},
  booktitle = {Advances in Neural Information Processing Systems},
  title     = {Sparse Weight Activation Training},
  url = {https://proceedings.neurips.cc/paper/2020/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf},
  month     = {December},
  year      = {2020},
}
​```
```
