#!/bin/bash
#SBATCH --account=rrg-franlp
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --time=0:59:00

export CUBLAS_WORKSPACE_CONFIG=:16:8
module load httpproxy
source ../ENV/bin/activate
python src/main.py --configs 'configs/resnet20_cifar10.jsonnet, configs/quantization/4bit_wcat.jsonnet, configs/attack/BFA.jsonnet' bfa