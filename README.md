# TensorFlow分布式训练
本次实验基于TensorFlow官方的例程来测试的分布式训练，
原代码[链接](https://github.com/tensorflow/models/tree/master/official/vision/image_classification)。

文件说明
- `resnet_cifar_main.py`以及`resnet_imagenet_main.py`:未修改的原代码
- `resnet_cifar_main_horovod.py`以及`resnet_imagenet_main_horovod.py`:支持horovod运行的代码。运行参考示例
  - `horovodrun --start-timeout 60 -np 16 -H 9.73.165.158:4,9.73.136.185:4,9.73.169.29:4,9.73.165.16:4 --verbose python resnet_cifar_main_horovod.py --data_dir cifar-10-batches-bin/ --distribution_strategy off`
  - `horovodrun --start-timeout 60 -np 16 -H 9.73.165.158:4,9.73.136.185:4,9.73.169.29:4,9.73.165.16:4 --verbose python resnet_imagenet_main_horovod.py --data_dir /dockerdata/tf_records/train/ --distribution_strategy off`
- `resnet_cifar_main_dist.py`以及`resnet_imagenet_main_dist.py`:支持Tensorflow distributed Strategy的代码，需要在不同机器上配置不同的`TF_CONFIG`。运行参考示例
  - `python resnet_cifar_main_dist.py --data_dir cifar-10-batches-bin/ --distribution_strategy multi_worker_mirrored --num_gpus 8`
  - `python resnet_cifar_imagenet_dist.py --data_dir /dockerdata/tf_records/train/ --distribution_strategy multi_worker_mirrored --num_gpus 8`

- ps_server文件夹下的文件是用来测试`TensorFlow`的`parameter_server`策略的，使用的时候，只需要在`run.sh`以及`kill.sh`中配置好ip地址以及文件，然后
  - `sh run.sh`：将一个gpu作为parameter server，剩余的15个gpu作为worker，这15个gpu运行程序的输出分别为log{1-15}.log。日志中包含了运行程序的时间等信息。
  - `sh kill.sh`：终止所有的`TensorFlow`代码


将整个项目克隆到本地\
`git clone https://github.com/tensorflow/models.git`\
我们实际的所需文件在这个路径下\
`PATH/model/official/vision/image_classification`\
这里的`PATH`根据你所处的路径而定。


## Part 0: 分布式训练的环境搭建
首先，我们需要在每一个机器上将基本的依赖包安装好，例如`Tensorflow-gpu 2.0`,`horovod`,`OpenMPI`等等。可以按照下面的步骤，检查或者安装所需的依赖包。

1. 将数据集以及models文件夹拷贝到每一台机器上。如果没有数据集，可以先跳过，最后将数据拷贝进去即可。
1. 确认`CUDA`版本为10.0（`TensorFlow-gpu 2.0`只能支持`CUDA 10.0`）
2. 确认`cudnn`已经安装，我用的是`cudnn 7.6`
   - `tar zxf cudnn-XXXX.tgz`
   - `cp cuda/include/cudnn.h /usr/local/cuda/include/`
   - `cp cuda/lib64/* /usr/local/cuda/lib64/`
   - `vim ~/.bashrc`
   - `export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/usr/local/cuda/bin/`
   - `export LD_LIBRARY_PATH=./:/usr/local/cuda/lib64:`
3. 安装代理，以便后续安装其他依赖包
   - `export http_proxy=10.223.1.32:3128`
   - `export https_proxy=10.223.1.32:3128`
4. 安装`anaconda3`环境，我用的是`python3.6.5`的`anaconda3`
5. 升级本地`pip` （`anaconda3`自动会将其`pip`和`python`设为默认）
   - `pip install --upgrade pip -i https://mirrors.cloud.tencent.com/pypi/simple`
6.  卸载原有的tensorflow-gpu，安装tf2.0rc版本
   - `pip uninstall tensorflow-gpu`
   - `pip install tensorflow-gpu==2.0.0rc -i https://mirrors.cloud.tencent.com/pypi/simple`
   - 如果有关于`wrapt`的报错，先执行下述命令，在安装tf。
   - `pip install -U --ignore-installed wrapt enum34 simplejson netaddr -i https://mirrors.cloud.tencent.com/pypi/simple`
7. 修改.bashrc 加入models路径, 否则后续运行代码时会出现不能导入包的错误
   - `vim ~/.bashrc`
   - `export PYTHONPATH="$PYTHONPATH:/root/models"`
   - `export PYTHONPATH="$PYTHONPATH:/root/official"`
   - `source ~/.bashrc`
8. 检查或者升级gcc版本, 需要gcc7.3.1。
   - `gcc --version`
   - `yum install centos-release-scl`
   - `yum install devtoolset-7-gcc*`
   - `yum install devtoolset-7-binutils`
   - `vim ~/.bashrc`
   - `. /opt/rh/devtoolset-7/enable`
9. 安装 OpenMPI，我这边用的时4.0.1
   - `wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz`
   - `gunzip -c openmpi-4.0.1.tar.gz | tar xf -`
   - `cd openmpi-4.0.1`
   - `./configure --prefix=/usr/local`
   - `make all install`
10. 安装horovod
    - `pip install horovod -i https://mirrors.cloud.tencent.com/pypi/simple`
    - 检查horovod.tensorflow，为了确保horovod.tensorflow安装成功，建议使用
    - `HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod -i https://mirrors.cloud.tencent.com/pypi/simple`

上述步骤如果都顺利完成，那么环境的搭建便完成了。

## Part I. 基于TensorFlow的Distributed Strategy的分布式训练
在TensorFlow 2.0RC中，`tf.distribute.Strategy`提供了一个分布式训练的接口。基于这个接口，可以在较小的修改代码的情况下完成分布式训练。TensorFlow目前支持几种常用策略，下面简单介绍一些这些策略，具体的介绍请参考官方的[Guide](https://www.tensorflow.org/beta/guide/distribute_strategy)。
1. `tf.distribute.MirroredStrategy`：这个策略是在**一个机器**上做利用多个gpu做**同步的**分布式训练的（`TensorFlow-gpu`如果不做任何指定，是在一个gpu上跑的）。这个策略在每一个gpu上都创建一个副本，每个变量都被“克隆”到每个gpu上。
2. `tf.distribute.experimental.MultiWorkerMirroredStrategy`：这个策略是在**多个机器**上利用多个gpu做**同步**的分布式训练。这个策略和`tf.distribute.MirroredStrategy`有点类似，只不过后者是在单个机器上。
3. `tf.distribute.experimental.TPUStrategy`:这个策略是用于TPU上的，因为了解的不多，不做介绍。
4. `tf.distribute.experimental.ParameterServerStrategy`:这个策略是在**多个机器**上利用多个gpu做**异步**的分布式训练。使用时需要指定`ParameterServer`和`worker`。这个策略在目前的tf2.0rc版本中并不完善。实际测试中，只能完成多机单卡的测例，如果要使用多机多卡，必须将每一个gpu都抽象成一个机器使用。


Strategy | 描述 |   
-|-
MirroredStrategy | 同步单机多卡 |
MultiWorkerMirroredStrategy | 同步多机多卡 |
ParameterServerStrategy | 异步多机多卡 |

策略一，也就是MirroredStrategy，在官方给的样例中测试十分简单，只需要执行的时候多加一个参数即可。\
`python resnet_cifar_main.py --data_dir cifar-10-batches-bin/ --num_gpus 3`
上述语句便可以在一台机器上利用多个显卡加速训练

下面我们介绍策略二和策略三的使用方法。

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

注意，这里在每个机器上都需要设置，并且每个机器的除了`index`对应的值不同，其余的都是一样的。因此，本步骤以及接下来的步骤在每个机器上都需要做一遍。也就是说，在实际跑分布式训练中，每台机器都需要运行同样的代码（除了`TF_CONFIG`设置略有不同）。


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

### 4. 在每个机器上运行代码
#### 4.1 同步MultiWorkerMirrored策略
`resnet_cifar_main.py`中已经留出了接口用于在输入的时候指定分布式的策略，最后只需要在每个机器上运行代码即可\
`python resnet_cifar_main.py --data_dir cifar-10-batches-bin/ --distribution_strategy multi_worker_mirrored --num_gpus 8`\
这里的几个参数的含义
1. `--data_dir`: 是数据集的路径
2. `--distribution_strategy`: 是分布式训练的策略，在多机多卡下为` multi_worker_mirrored`
3. `--num_gpus`: 表示用的gpu个数

接下来就是training的过程了。

#### 4.2 异步ParameterServer策略
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

如果需要实现多机多卡的异步训练，目前的方法是需要将`TF_CONFIG`设置成下述格式
```
import os
import json
os.environ["TF_CONFIG"] = json.dumps({
      'cluster': {
        'ps' : ["100.102.32.179:8080"],
        'worker': ["100.102.33.40:8080","100.102.33.40:8081"] # 相同的ip，不同的端口
                  },
        'task': # {'type':'ps', 'index':0}
            {'type': 'worker', 'index': 1} # 可以将每一个端口看成一个worker
          })
```
然后需要在每个机器开多个终端，每个终端的只能有一个gpu处于可见状态，并且每个终端的可见gpu都不一样。接下来和之前的训练一样，只需要在每个终端上运行代码即可。

## Part II. 基于Horovod的分布式训练
Horovod是一个分布式训练的框架，它支持`TensorFlow`, `keras`, `PyTorch`以及`MXNet`。基于Horovod，只需要较小的改动代码即可实现分布式的训练。

### 1. 安装Horovod
官方给的安装流程可以参考[link](https://github.com/horovod/horovod#install)。
1. 安装Open MPI：如果按官方的guide安装有问题，可以参考[link](https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX)
2. 确认gcc版本：这边明确一下自己的`TensorFlow`版本，官方目前给的建议是`g++-4.85`，但是如果你用的是`Tensorflow 2.0`，实际发现需要`g++-7.3.1`。否则后续安装Horovod会出bug。
3. 安装Horovod: `pip install horovod`。建议安装完后测试一下`import horovod.tensorflow`，如果不报错，那么就已经安装完成了。如果报错，检查一下第二步是否出了问题。

安装完成之后，可以测试一下`mpirun`和`horovodrun`是否可以用。如果`mpirun`没有这个指令，那么很可能是没有将mpirun加入到路径中。
```
export PATH=YOUR_PATH_FOR_MPI/bin:$PATH
```
如果`horovodrun`不能正常使用，很有可能是因为调用的python解释器的问题。`horovodrun`默认是以`#!/usr/bin/python`去作为解释器的，但是实际中可能用的`anaconda`或者是用户本地的`python`。因此应该将`horovodrun`的第一句话改成你所希望的解释。`horovodrun`在我这里是位于`~/anaconda3/bin`里面。
### 2. 修改代码
安装完成之后，则需要修改部分代码。可以参考[代码](https://github.com/horovod/horovod/blob/master/examples/tensorflow2_keras_mnist.py)修改。


### 3. 下载cifar数据集
后面几步和Part I类似，如果已经完成，则无需再做。
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
至此，数据集便准备完毕

### 4. 关闭所有代理
实际测试中发现，代理的存在会使得连接无法建立，因此，为确保连接能够建立，清空所有代理设置。
```
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```
### 5. 运行代码
horovod不需要在每个机器上都去跑一个代码了，这边只需要保证两台机器需要运行的文件在相同的路径下。只需要在一台机器是使用`horovodrun --verbose -np 4 -H 100.102.32.179:2,100.102.33.44:2 python resnet_cifar_main.py --data_dir cifar-10-batches-bin/ --distribution_strategy off`。这里有几点说明。
1. `-H`后面跟的是`ip:port,ip:port`，其中间不能有空格，也就是不能出现`ip:port,[空格]ip:port`。否则会解析失败。
2. 保证两边的`hostname`是不一样的。
3. `-np`说明了一共用多少gpu跑，当前这句话的意思是，一共用4个gpu, 在100.102.32.179上用2个, 在100.102.33.44上用2个。

## Imagenet训练代码
除了数据集不一样以外，剩下的内容都和之前的一样。Imagenet的数据集需要以tf_records的格式输入，参考下述方法生成。
1. tf_records文件的生成参考[link](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy)。官方给的代码需要用`python2`和`tf 1.X`版本去执行。

数据集搞定之后，剩下的内容和之前一样。

`horovodrun --start-timeout 60 -np 16 -H 100.102.33.44:8,100.102.32.179:8 --verbose python resnet_imagenet_main_horovod.py --data_dir ~/imageNet_2012/tf_records/train/ --distribution_strategy off`

1. 设置`--start-timeout 60`原因是测试机器ssh连接较慢，需要更多等待时间。


## Part IV 测试结果
我们测试了在遍历了一整个imagenet数据所需的时间，`batch_size = 192`

### horovod 测试结果

设置|耗时|
-|-|
单机单卡|7299s|
单机多卡（4个gpu）|2143s|
多机多卡 （4台机器，每台4个gpu）|897s|

### parameter_server测试结果
server|ps server index| cost time|
-|-|-|
server 0(local)|0 |

server|worker index| cost time|
-|-|-|
server 0 (local)|0|652|
server 0 (local)|1|665|
server 0 (local)|2|678|
server 1 (remote)|3|929|
server 1 (remote)|4|1008|
server 1 (remote)|5|999|
server 1 (remote)|6|996|
server 2 (remote)|7|760|
server 2 (remote)|8|668|
server 2 (remote)|9|679|
server 2 (remote)|10|682|
server 3 (remote)|11|708|
server 3 (remote)|12|725|
server 3 (remote)|13|779|
server 3 (remote)|14|762|

average time| maximum time
-|-|
779.3|1008
