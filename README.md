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
            * [Cifar10](#cifar10)
            * [Cifar100](#cifar100)
            * [ImageNet](#imagenet)
         * [Structured SWAT](#structured-swat)
            * [Cifar10](#cifar10-1)
            * [Cifar100](#cifar100-1)
            * [ImageNet](#imagement)
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

*Running Inference:* 

```python main.py -model "ResNet18" -dataset="Cifar10"  --schedule-file $1 --inference 1 --checkpoint $2   ```

Schedule-files are present in run_configurations [directory](https://github.com/AamirRaihan/SWAT/tree/main/SWAT-code/cifar10-100-code/run_configurations).

*Running Training:* 

```./run_script/resnet_cifar10.sh run_configurations/unstructured_constant_resnet18_schedule_90.yaml ```

|   Network    |  Method  | Weight Sparsity (%) | Activation Sparsity (%) | Top-1 Accuracy (%) | Training FLOP ⬇ % | Checkpoint                                                   |
| :----------: | :------: | :-----------------: | :---------------------: | :----------------: | :---------------: | ------------------------------------------------------------ |
|    VGG-16    |  SWAT-U  |        90.0         |          90.0           |     91.95±0.06     |       89.7        | [here](https://drive.google.com/drive/u/1/folders/1sW8pMzUQoNY4lnTia2MspAtIoNXZHAqu) |
|    VGG-16    | SWAT-ERK |        95.0         |          82.0           |     92.50±0.07     |       89.5        | [here](https://drive.google.com/drive/u/1/folders/1GMmOStXCNCeLHr2UMYKOLVGx6kKJ9PLL) |
|    VGG-16    |  SWAT-M  |        95.0         |          65.0           |     93.41±0.05     |       64.0        | [here](https://drive.google.com/drive/u/1/folders/1-NSGNKXmPCmchMt5sDTMK23hRGm37Je_) |
|   WRN-16-8   |  SWAT-U  |        90.0         |          90.0           |     95.13±0.11     |       90.0        | [here](https://drive.google.com/drive/u/1/folders/1Rcs5n1NeRiypaL-S3tV0r8bJSapJ-Zh7) |
|   WRN-16-8   | SWAT-ERK |        95.0         |          84.0           |     95.00±0.12     |       91.4        | [here](https://drive.google.com/drive/u/1/folders/1Bxs-w3UQIDL23sYpfQwkavLkVWTApHI4) |
|   WRN-16-8   |  SWAT-M  |        95.0         |          78.0           |     94.97±0.04     |       86.3        | [here](https://drive.google.com/drive/u/1/folders/19VHi7WFsaa7DyJZQ2AEKcaeiMeoj-JI-) |
| DenseNet-121 |  SWAT-U  |        90.0         |          90.0           |     94.48±0.06     |       89.8        | [here](https://drive.google.com/drive/u/1/folders/18wdBoaBtyLRZI75Z2itTMs2eI-JGJCj1) |
| DenseNet-121 | SWAT-ERK |        90.0         |          88.0           |     94.14±0.11     |       89.7        | [here](https://drive.google.com/drive/u/1/folders/1YrMrxe9CCzJZ3AmXvSdTFa_KWyScE6kB) |
| DenseNet-121 |  SWAT-M  |        90.0         |          86.0           |     94.29±0.11     |       84.2        | [here](https://drive.google.com/drive/u/1/folders/1NcyvGcnt8XM5LK2sup9TWl8gfRJP3IAG) |

Note more checkpoints are available [here](https://drive.google.com/drive/folders/148nfMxjhn5_vLVBqJe-z2L_vRL7-tohO?usp=sharing). Basically, the checkpoints for different sparsity percentage are present there. Moreover, ResNet-18 data is also present. You can also plot the train/test and loss curve for all the individual training run using the data present in the directory.

#### CIFAR100

*Running Inference:* 

```python main.py -model "ResNet18" -dataset="Cifar100"  --schedule-file $1 --inference 1 --checkpoint $2   ```

Schedule-files are present in run_configurations [directory](https://github.com/AamirRaihan/SWAT/tree/main/SWAT-code/cifar10-100-code/run_configurations).

*Running Training:* 

```./run_script/resnet_cifar100.sh run_configurations/unstructured_constant_resnet18_schedule_90.yaml ```

|   Network    |  Method  | Weight Sparsity(%) | Activation Sparsity(%) | Top-1 Accuracy(%) | Training FLOP ⬇ % | Checkpoint                                                   |
| :----------: | :------: | :----------------: | :--------------------: | :---------------: | :---------------: | ------------------------------------------------------------ |
|    VGG-16    |  SWAT-U  |        90.0        |          90.0          |    91.95±0.08     |         -         | [here](https://drive.google.com/drive/folders/1vk8dNB2RxQz695buv43Lyh3e1fhS2xFd?usp=sharing) |
|    VGG-16    | SWAT-ERK |        90.0        |          69.6          |    92.50±0.11     |         -         | [here](https://drive.google.com/drive/folders/1GRc46x7SKPiYDTw9FZqINocOI2mpo2-R?usp=sharing) |
|    VGG-16    |  SWAT-M  |        90.0        |          59.9          |    93.41±0.23     |         -         | [here](https://drive.google.com/drive/folders/1uOMsPcVWjoz6KvgyMdorKnZfAbXGo_7V?usp=sharing) |
|   WRN-16-8   |  SWAT-U  |        90.0        |          90.0          |    95.13±0.13     |         -         | [here](https://drive.google.com/drive/folders/1OqnDwn-A7U83pYIHCbOtePW-r-ZW5t9e?usp=sharing) |
|   WRN-16-8   | SWAT-ERK |        90.0        |          77.6          |    95.00±0.07     |         -         | [here](https://drive.google.com/drive/folders/1xmEXQonCkdHvh6yWo7T7RqnS3_8GqVO4?usp=sharing) |
|   WRN-16-8   |  SWAT-M  |        90.0        |          73.3          |    94.97±0.11     |         -         | [here](https://drive.google.com/drive/folders/15Mc9zFc6YCJTuFc2ZUsG2Wyo1Ywhs9eJ?usp=sharing) |
| DenseNet-121 |  SWAT-U  |        90.0        |          90.0          |    94.48±0.06     |         -         | [here](https://drive.google.com/drive/folders/1NiVUpFqKxmX5ff_sv6YTDeEk9zWEAE1F?usp=sharing) |
| DenseNet-121 | SWAT-ERK |        90.0        |          90.0          |    94.14±0.03     |         -         | [here](https://drive.google.com/drive/folders/1BZkO_zr7y9b5Ofyv5wqfMyJ0CEoyu6yM?usp=sharing) |
| DenseNet-121 |  SWAT-M  |        90.0        |          84.2          |    94.29±0.13     |         -         | [here](https://drive.google.com/drive/folders/1BZkO_zr7y9b5Ofyv5wqfMyJ0CEoyu6yM?usp=sharing) |

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

*Running Inference:* 

```python main.py -model "ResNet18" -dataset="Cifar10"  --schedule-file $1 --inference 1 --checkpoint $2   ```

Schedule-files are present in run_configurations [directory](https://github.com/AamirRaihan/SWAT/tree/main/SWAT-code/cifar10-100-code/run_configurations).

*Running Training:* 

```./run_script/resnet_cifar10.sh run_configurations/structured_constant_resnet18_schedule_70.yaml ```

|   Network    | Method | Weight Sparsity (%) | Activation Sparsity (%) | Channel Pruned (%) | Top-1 Accuracy (%) | Training FLOP ⬇ % | Checkpoint                                                   |
| :----------: | :----: | :-----------------: | :---------------------: | :----------------: | :----------------: | :---------------: | ------------------------------------------------------------ |
|  ResNet-18   | SWAT-U |        50.0         |          50.0           |        50.0        |     94.73±0.06     |       49.9        | [here](https://drive.google.com/drive/folders/1qexXzrHQiCFo4UCpMJHmy1pZmcDfO_Rt?usp=sharing) |
|  ResNet-18   | SWAT-U |        60.0         |          60.0           |        60.0        |     94.68±0.03     |       59.8        | [here](https://drive.google.com/drive/folders/16W4oMfEuIOdAx67jX26zF98gnWJtessy?usp=sharing) |
|  ResNet-18   | SWAT-U |        70.0         |          70.0           |        70.0        |     94.65±0.19     |       69.8        | [here](https://drive.google.com/drive/folders/1OhV9aDiZSs32HO0S1bJtvgPOwTNdSWtW?usp=sharing) |
| DenseNet-121 | SWAT-U |        50.0         |          50.0           |        50.0        |     95.04±0.26     |       49.9        | [here](https://drive.google.com/drive/folders/1BTB6rM64Yvv2bjy0RymerWQ4MZL-wT5m?usp=sharing) |
| DenseNet-121 | SWAT-U |        60.0         |          60.0           |        60.0        |     94.82±0.11     |       59.9        | [here](https://drive.google.com/drive/folders/1R7knbrCrmmA3DbNzndOKiMAer9q4uiIc?usp=sharing) |
| DenseNet-121 | SWAT-U |        70.0         |          70.0           |        70.0        |     94.81±0.20     |       69.9        | [here](https://drive.google.com/drive/folders/1afGv4Fo1lkz4w-ohUxQv2HcS-tXYflMi?usp=sharing) |

#### CIFAR100

*Running Inference:* 

```python main.py -model "ResNet18" -dataset="Cifar100"  --schedule-file $1 --inference 1 --checkpoint $2   ```

Schedule-files are present in run_configurations [directory](https://github.com/AamirRaihan/SWAT/tree/main/SWAT-code/cifar10-100-code/run_configurations).

*Running Training:* 

```./run_script/resnet_cifar100.sh run_configurations/structured_constant_resnet18_schedule_70.yaml ```

|   Network    | Method | Weight Sparsity (%) | Activation Sparsity (%) | Channel Pruned (%) | Top-1 Accuracy (%) | Checkpoint                                                   |
| :----------: | :----: | :-----------------: | :---------------------: | :----------------: | :----------------: | ------------------------------------------------------------ |
|  ResNet-18   | SWAT-U |        50.0         |          50.0           |        50.0        |     76.4±0.05      | [here](https://drive.google.com/drive/folders/1RowHtd3C4MY7MupoaYpOcM_MlzBO2DS9?usp=sharing) |
|  ResNet-18   | SWAT-U |        60.0         |          60.0           |        60.0        |     76.2±0.11      | [here](https://drive.google.com/drive/folders/1PlGtItABO7NzhtTK-HabnD1YrKt2Gv0U?usp=sharing) |
|  ResNet-18   | SWAT-U |        70.0         |          70.0           |        70.0        |     75.6±0.09      | [here](https://drive.google.com/drive/folders/1vM97Dsn4C5ZYi1BuVjJMRCHgko6sn7pR?usp=sharing) |
| DenseNet-121 | SWAT-U |        50.0         |          50.0           |        50.0        |     78.7±0.03      | [here](https://drive.google.com/drive/folders/1vM97Dsn4C5ZYi1BuVjJMRCHgko6sn7pR?usp=sharing) |
| DenseNet-121 | SWAT-U |        60.0         |          60.0           |        60.0        |     78.5±0.08      | [here](https://drive.google.com/drive/folders/1XGPM2l320WGkSRC55a8OxZoZpSWhmizG?usp=sharing) |
| DenseNet-121 | SWAT-U |        70.0         |          70.0           |        70.0        |     78.1±0.12      | [here](https://drive.google.com/drive/folders/1Fl9mWRUQo2ygCEQ5Xh1IK7qmfunimlN1?usp=sharing) |

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

1. **Unstructured-SWAT on CIFAR-10 Dataset :** [here](https://drive.google.com/drive/folders/1ej5Sg4KVb-5y7b7D_d7HH7gp1rHyzKGp?usp=sharing)
2. **Unstructured-SWAT on CIFAR-100 Dataset :** [here](https://drive.google.com/drive/folders/1sScgmq8LlFaM7Ylo9IJi2PSlifVlZpVj?usp=sharing)
3. **Structured-SWAT on CIFAR-10 Dataset :** [here](https://drive.google.com/drive/folders/1pGYuwH24yfxWoKbf7KVX5iBq_Z2nGg3V?usp=sharing)
4. **Structured-SWAT on CIFAR-100 Dataset :** [here](https://drive.google.com/drive/folders/1xjjZ-tV9t4FvUnV694rptTFUCTrffAfZ?usp=sharing)

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
