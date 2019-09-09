#! /bin/bash

for i in $(seq 0 3)
do
ssh root@9.73.165.158  python /root/models/official/vision/image_classification/ps_server/resnet_imagenet_main_dist_ps_$i.py --data_dir /dockerdata/tf_records/train/ --distribution_strategy parameter_server --batch_size 192 --train_epochs 1 2>log$i.log 1>/dev/null &
#echo $cmd
sleep 1
done

for i in $(seq 4 7)
do
ssh root@9.73.136.185  python /root/models/official/vision/image_classification/ps_server/resnet_imagenet_main_dist_ps_$i.py --data_dir /dockerdata/tf_records/train/ --distribution_strategy parameter_server --batch_size 192 --train_epochs 1 2>log$i.log 1>/dev/null &
#echo $cmd
sleep 1
done

for i in $(seq 8 11)
do
ssh root@9.73.169.29  python /root/models/official/vision/image_classification/ps_server/resnet_imagenet_main_dist_ps_$i.py --data_dir /dockerdata/tf_records/train/ --distribution_strategy parameter_server --batch_size 192 --train_epochs 1 2>log$i.log 1>/dev/null &
sleep 1
#echo $cmd
done

for i in $(seq 12 15)
do
ssh root@9.73.165.16  python /root/models/official/vision/image_classification/ps_server/resnet_imagenet_main_dist_ps_$i.py --data_dir /dockerdata/tf_records/train/ --distribution_strategy parameter_server --batch_size 192 --train_epochs 1 2>log$i.log 1>/dev/null &
#echo $cmd
sleep 1
done
