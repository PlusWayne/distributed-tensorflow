本次实验基于TensorFlow官方的例程来测试多机多卡的分布式训练。

代码位于[link](https://github.com/tensorflow/models/tree/master/official/vision/image_classification)。

首先将整个项目克隆到本地\
`git clone https://github.com/tensorflow/models.git`\
我们实际的所需文件在这个路径下\
`PATH/model/official/vision/image_classification`\
基于该目录下的`resnet_cifar_main.py`来进行分布式训练。

> 这里需要把PATH/model加入到路python径中，也就是要执行如下命令\
`export PYTHONPATH="$PYTHONPATH:/data1/wayneweixu/models"`


# 配置TF_CONFIG环境变量
分布式的TensorFlow的训练首先需要在每个机器上配置好`TF_CONFIG`环境变量，用来指定每个一个节点的用途。官方给的方法是通过如下方式设置的
```
import os
import json
os.environ["TF_CONFIG"] = json.dumps({
        'cluster': {
          'worker': ["100.102.32.179:8080", "100.102.33.40:8080"]
                   },
            'task': {'type': 'worker', 'index': 0}
          })
```
这里`worker`需要给出所有的节点的`ip:port`。注意，这里不需要用户名，只需要ip地址即可。`task`是指定当前node的作用，如果其`index=0`，那么它就作为`chief worker`。
我们只需要将上述代码**加到**`resnet_cifar_main.py`中的开头即可。
>注意，这里在每个机器上都需要设置，并且每个机器的除了`index`对应的值不同，其余的都是一样的。因此，本步骤以及接下来的步骤在每个机器上都需要做一遍。

# 下载cifar数据集
可以利用项目中的脚本下载\
`python ../../r1/resnet/cifar10_download_and_extract.py`\
也可以自己从网上下载并解压缩到当前目录。解压缩完毕后，会有如下的文件\
cifar-10-batches-bin/\
├── batches.meta.txt\
├── data_batch_1.bin\
├── data_batch_2.bin\
├── data_batch_3.bin\
├── data_batch_4.bin\
├── data_batch_5.bin\
├── readme.html\
└── test_batch.bin\
至此，数据集便准备完毕了


# 关闭所有代理
代理的存在会使得连接无法建立，因此，为确保连接能够建立，需要将所有代理设置为空
```
export http_proxy=''
export https_proxy=''
export HTTP_PROXY=''
export HTTPS_PROXY=''
```

# 在每个机器上运行代码
最后只需要在每个机器上运行代码即可\
`python resnet_cifar_main.py --data_dir cifar-10-batches-bin/ --distribution_strategy multi_worker_mirrored --num_gpus 8`\
这里的几个参数的含义
1. `--data_dir`是数据集的路径
2. `--distribution_strategy`是分布式训练的策略，在多机多卡下为` multi_worker_mirrored`
3. `--num_gpus`表示用的gpu个数

接下来就是training的过程了。
