#! /bin/bash
export http_proxy=''
export https_proxy=''
export HTTP_PROXY=''
export HTTPS_PROXY=''

export PYTHONPATH="$PYTHONPATH:/data1/wayneweixu/models"

python resnet_cifar_main_dist.py --data_dir cifar-10-batches-bin/ --distribution_strategy multi_worker_mirrored --num_gpus 8
#python resnet_cifar_main_dist.py --data_dir cifar-10-batches-bin/ --distribution_strategy parameter_server
