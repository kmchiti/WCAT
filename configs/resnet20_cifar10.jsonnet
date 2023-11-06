local base_config = (import 'base.jsonnet');
local model_config = (import 'model/resnet20.jsonnet');
local dataset_config = (import 'dataset/cifar10.jsonnet');
local train_config = (import 'trainer/cifar_config.jsonnet');

model_config + dataset_config + train_config + base_config