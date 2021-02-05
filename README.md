# Undistillable: Making A Nasty Teacher That CANNOT teach students
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

["Undistillable: Making A Nasty Teacher That CANNOT teach students"](https://openreview.net/forum?id=0zvfm-nZqQs)

Haoyu Ma, Tianlong Chen, Ting-Kuei Hu, Chenyu You, Xiaohui Xie, Zhangyang Wang    
In ICLR 2021 Spotlight Oral



## Overview 

* We propose the concept of **Nasty Teacher**, a defensive approach to prevent knowledge leaking and unauthorized model cloning through KD without sacrificing performance. 
* We propose a simple yet efficient algorithm, called **self-undermining knowledge distillation**, to directly build a nasty teacher through self-training, requiring no additional dataset
nor auxiliary network. 


## Prerequisite
We use Pytorch 1.4.0, and CUDA 10.1. You can install them with  
~~~
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
~~~   
It should also be applicable to other Pytorch and CUDA versions.  


Then install other packages by
~~~
pip install -r requirements.txt
~~~

## Usage 


### Teacher networks 

##### Step 1: Train a normal teacher network   

~~~
python train_scratch.py --save_path [XXX]
~~~
Here, [XXX] specifies the directory of `params.json`, which contains all hyperparameters to train a network.
We already include all hyperparameters in `experiments` to reproduce the results in our paper.    

For example, normally train a ResNet18 on CIFAR-10  
~~~
python train_scratch.py --save_path experiments/CIFAR10/baseline/resnet18
~~~
After finishing training, you will get `training.log`, `best_model.tar` in that directory.  
   
The normal teacher network will serve as the **adversarial network** for the training of the nasty teacher. 



##### Step 2: Train a nasty teacher network
~~~
python train_nasty.py --save_path [XXX]
~~~
Again, [XXX] specifies the directory of `params.json`, 
which contains the information of adversarial networks and hyperparameters for training.  
You need to specify the architecture of adversarial network and its checkpoint in this file. 

 
For example, train a nasty ResNet18
~~~
python train_nasty.py --save_path experiments/CIFAR10/kd_nasty_resnet18/nasty_resnet18
~~~


### Knowledge Distillation for Student networks 

You can train a student distilling from normal or nasty teachers by 
~~~
python train_kd.py --save_path [XXX]
~~~
Again, [XXX] specifies the directory of `params.json`, 
which contains the information of student networks and teacher networks
 

For example,   
* train a plain CNN distilling from a nasty ResNet18 
~~~
python train_kd.py --save_path experiments/CIFAR10/kd_nasty_resnet18/cnn
~~~

* Train a plain CNN distilling from a normal ResNet18 
~~~
python train_kd.py --save_path experiments/CIFAR10/kd_normal_resnet18/cnn
~~~



## Citation
~~~
@inproceedings{
ma2021undistillable,
title={Undistillable: Making A Nasty Teacher That {\{}CANNOT{\}} teach students},
author={Haoyu Ma and Tianlong Chen and Ting-Kuei Hu and Chenyu You and Xiaohui Xie and Zhangyang Wang},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=0zvfm-nZqQs}
}
~~~

## Acknowledgement
* [Teacher-free KD](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation)
* [DAFL](https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL) 
* [DeepInversion](https://github.com/NVlabs/DeepInversion)

