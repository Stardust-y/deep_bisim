# bisim环境配置
- ubuntu18.04虚拟机
- anaconda3 5.2.0
## 第一步
    conda env create -f conda_env.yml
## 第二步
    $ python train.py --domain_name cheetah --task_name run --encoder_type pixel --decoder_type identity --action_repeat 4 --save_video --save_tb --work_dir ./log --seed 1
## 运行时报错
    AttributeError:
### 解决
    $ pip uninstall torchvision
    $ pip install torchvision==0.7.0
    $ pip install --no-deps torchvision==0.7.0
### 补充安装
    pip install opencv-python
    pip install scikit-video/tqdm/agents
### 报错
    ImportError: TensorBoard logging requires TensorBoard with Python summary writer installed. This should be available in 1.14 or above
### 解决
    conda install -c conda-forge tensorboard
    conda install -c conda-forge protobuf
### 报错
    TypeError: __init__() got an unexpected keyword argument 'serialized_options'
### 解决
    pip install -U protobuf
### 报错
    AttributeError: module 'tensorflow' has no attribute 'contrib'
### 解决
    pip install tensorflow==1.9.0
### 报错
    OSError: Cannot find MuJoCo library at
### 解决
    在官网山安装mujoco,并解压到C:/users/name/.mujoco下
### 报错
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
### 解决
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
### 报错
    dm_control.mujoco.wrapper.core.Error: Could not register license.
### 解决
    在./mojoco路径下添加mjkey.txt
### 报错
    需要51.9GB内存
### 解决
    设置命令行参数 replay_buffer_capacity 1000
### 报错
    AttributeError: module 'tensorflow' has no attribute 'io'
### 解决
    pip install tensorflow-io
