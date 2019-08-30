# TensorFlow分布式训练
本次实验基于TensorFlow官方的例程来测试的分布式训练。

代码位于[link](https://github.com/tensorflow/models/tree/master/official/vision/image_classification)。
## Part I. 基于TensorFlow的Distributed Strategy的分布式训练
在TensorFlow 2.0RC中，`tf.distribute.Strategy`提供了一个分布式训练的接口。基于这个接口，可以在较小的修改代码的情况下完成分布式训练。TensorFlow目前支持几种常用策略，下面简单介绍一些这些策略，具体的介绍请参考官方的[Guide](https://www.tensorflow.org/beta/guide/distribute_strategy)。
1. `tf.distribute.MirroredStrategy`：这个策略是在**一个机器**上做利用多个gpu做**同步的**分布式训练的（`TensorFlow-gpu`如果不做任何指定，是在一个gpu上跑的）。这个策略在每一个gpu上都创建一个副本，每个变量都被“克隆”到每个gpu上。
2. `tf.distribute.experimental.MultiWorkerMirroredStrategy`：这个策略是在**多个机器**上利用多个gpu做**同步**的分布式训练。这个策略和`tf.distribute.MirroredStrategy`有点类似，只不过后者是在单个机器上。
3. `tf.distribute.experimental.TPUStrategy`:这个策略是用于TPU上的，因为了解的不多，不做介绍。
4. `tf.distribute.experimental.ParameterServerStrategy`:这个策略是在**多个机器**上利用多个gpu做**异步**的分布式训练。使用时需要指定`ParameterServer`和`worker`。这个策略在tf2.0rc版本中仅支持有限的几个场景。

下面我们用实际的例子来测试多机多卡的同步分布式训练，也就是策略2。

首先将整个项目克隆到本地\
`git clone https://github.com/tensorflow/models.git`\
我们实际的所需文件在这个路径下\
`PATH/model/official/vision/image_classification`\
这里的`PATH`根据你所处的路径而定。

我们基于该目录下的`resnet_cifar_main.py`来进行分布式训练。

> 这里需要把PATH/model加入到路python径中，也就是要执行如下命令\
`export PYTHONPATH="$PYTHONPATH:PATH/models"`


### 1. 配置TF_CONFIG环境变量
对于多个机器的分布式训练，在每个机器上，都需要配置好`TF_CONFIG`环境变量，用来说明每一台机器在分布式训练中承担的角色。官方给的方法是通过如下方式设置的
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
我们只需要将上述代码**加到**`resnet_cifar_main.py`中的开头即可。\
另外，如果没有多台机器的话，也是可以在本地做测试的，此时只需要把`ip:port`设置为`localhost:8080`以及`localhost:8081`。然后启动两个终端，分别在不同的终端运行代码即可。不过有可能需要在每个终端界面指定不同的gpu，否则有可能会出现gpu内存分配失败的错误，也就是在不同的终端，执行
```
export CUDA_VISIBLE_DEVICES=1,2,3 # terminal 1
export CUDA_VISIBLE_DEVICES=4,5,6 # terminal 2
```
>注意，这里在每个机器上都需要设置，并且每个机器的除了`index`对应的值不同，其余的都是一样的。因此，本步骤以及接下来的步骤在每个机器上都需要做一遍。也就是说，在实际跑分布式训练中，每台机器都需要运行同样的代码（除了`TF_CONFIG`设置略有不同）。

### 2. 下载cifar数据集
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


### 3. 关闭所有代理
实际测试中发现，代理的存在会使得连接无法建立，因此，为确保连接能够建立，清空所有代理设置。
```
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```

# 4. 在每个机器上运行代码
`resnet_cifar_main.py`中已经留出了接口用于在输入的时候指定分布式的策略，最后只需要在每个机器上运行代码即可\
`python resnet_cifar_main.py --data_dir cifar-10-batches-bin/ --distribution_strategy multi_worker_mirrored --num_gpus 8`\
这里的几个参数的含义
1. `--data_dir`: 是数据集的路径
2. `--distribution_strategy`: 是分布式训练的策略，在多机多卡下为` multi_worker_mirrored`
3. `--num_gpus`: 表示用的gpu个数

接下来就是training的过程了。

### 异步ParameterServer策略
`tf2.0`对`tf.distribute.experimental.ParameterServerStrategy`的支持还是很有限的，但是设置其实大部分是相同的，简单的说，只需要在每台机器把`TF_CONFIG`设置成如下形式即可
```
import os
import json
os.environ["TF_CONFIG"] = json.dumps({
      'cluster': {
        'ps' : ["100.102.32.179:8080"],
        'worker': ["100.102.33.40:8080"]
                  },
        'task': # {'type':'ps', 'index':0} # if the node is ParameterServer
            {'type': 'worker', 'index': 0} # if the node is worker
          })
```
然后在运行的时候，把策略重新指定下`python resnet_cifar_main.py --data_dir cifar-10-batches-bin/ --distribution_strategy parameter_server`
