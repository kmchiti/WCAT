from .base_model import Base_Model

############## ResNet for CIFAR-10 ###########
from .resnet import Resnet20, Resnet32, Resnet44, Resnet56

############## Lenet for MNIST #############
from .lenet import LeNet, Simple_MLP

############## ResNet for ImageNet #############
from .resnet_imagenet import Resnet18, Resnet34, Resnet50


############## VIT for ImageNet #############
from .vit import DeiT_Tiny_Patch16_224, DeiT_Small_Patch16_224, DeiT_Base_Patch16_224
