# Training DNNs Resilient to Adversarial and Random Bit-Flips by Learning Quantization Ranges

This repository contains a PyTorch implementation of weight clipping-aware training (WCAT) as described in our TMLR-2023 [paper](https://openreview.net/forum?id=BxjHMPwZIH&).

Abstract:

Promoting robustness in deep neural networks (DNNs) is crucial for their reliable deployment in uncertain environments, such as low-power settings or in the presence of adversarial attacks. In particular, bit-flip weight perturbations in quantized networks can significantly degrade performance, underscoring the need to improve DNN resilience. In this paper, we introduce a training mechanism to learn the quantization range of different DNN layers to enhance DNN robustness against bit-flip errors on the model parameters. The proposed approach, called weight clipping-aware training (WCAT), minimizes the quantization range while preserving performance, striking a balance between the two. 
Our experimental results on different models and datasets showcase that DNNs trained with WCAT can tolerate a high amount of noise while keeping the accuracy close to the baseline model. Moreover, we show that our method significantly enhances DNN robustness against adversarial bit-flip attacks. Finally, when considering the energy-reliability trade-off inherent in on-chip SRAM memories, we observe that WCAT consistently improves the Pareto frontier of test accuracy and energy consumption across diverse models.

## Quick Start
### Install requirements
Python: 3.8+
```setup
pip install -r requirements.txt 
```

### Training
To train a model on a given dataset using a specific quantization method, modify the configuration in the following command:
```setup
python src/main.py --configs 'configs/resnet20_cifar10.jsonnet, configs/quantization/4bit_wcat.jsonnet' train
```
Please ensure that you have the necessary dataset and pre-trained models for this step.

### Evaluation
Use the same configuration for the model, training recipe, dataset, and quantization to evaluate the model's performance against various attacks:

For random bit flip:
```setup
python src/main.py --configs 'configs/resnet20_cifar10.jsonnet, configs/quantization/4bit_wcat.jsonnet, configs/attack/random_flip.jsonnet' random_flip
```
For Bit-Flip Attack:
```setup
python src/main.py --configs 'configs/resnet20_cifar10.jsonnet, configs/quantization/4bit_wcat.jsonnet, configs/attack/BFA.jsonnet' bfa
```

## Citation
If you use this code for your research, please consider citing our paper:
```setup
@article{
chitsaz2023training,
title={Training DNNs Resilient to Adversarial and Random Bit-Flips by Learning Quantization Ranges},
author={Kamran Chitsaz and Goncalo Mordido and Jean-Pierre David and Francois Leduc-Primeau},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=BxjHMPwZIH},
note={}
}
```
