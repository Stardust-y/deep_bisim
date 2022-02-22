import sys
import os
import threading
import time
from typing import Callable
# 获取当前文件路径
import torch
import argparse
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
print(father_path)
sys.path.append(father_path)    # /home/yuxuehui/yxhfile/causal-metarl-master/src
father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
sys.path.append(father_path)    # /home/yuxuehui/yxhfile/causal-metarl-master
print(father_path)

import numpy as np

# from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from src_HER_level3.env_do_all_her_2reward import CBNEnv
from src.a2c import A2C
# from src_HER_linear.default import default_config


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        # print("**********************self.locals", self.locals)
        self.logger.record('random_value', value)
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train(args):
    current_path = os.path.abspath(__file__)
    father_path1 = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    father_path = os.path.abspath(os.path.dirname(father_path1) + os.path.sep + ".")
    print(args.vertex, args.action_range, args.model_dir)
    env = CBNEnv.create(
        info_phase_length=1440,
        action_range=args.action_range,
        vertex=args.vertex,
        # reward_weight=args.reward_weight,
        reward_scale=args.reward_scale,
        list_last_vertex=[
            # {   # 前1阶段
            #     "vertex":[6,7],
            #     "dir":father_path + "/Model_CHO2/her_low_action_67_goal_3_mlp_range_0_600_end_30_meal40_2.zip",
            #     "info_phase_length": 1440,
            #     "action_range":[0,600],
            #     "reward_scale": [1.0, 1.0],
            #     "last_vertex":[3]  },
            # {   # 前2阶段
            #     "vertex":[3],
            #     "dir":father_path + "/Model/her_low_action_3_goal_12_2reward3_copy2.zip",
            #     "info_phase_length": 1440,
            #     "action_range":[-np.inf, np.inf],
            #     "reward_scale":[1.0, 1.0],
            #     "last_vertex":[12]  }
        ],
        n_env=1
    )
    optimizer_kwargs = dict(
        alpha=0.95,
    )
    policy_kwargs = dict(
        optimizer_kwargs=optimizer_kwargs,
        optimizer_class=RMSpropTFLike
    )
    model = A2C("MlpPolicy",
                env,
                gamma=0.93,  # 折扣因子，计算 discounted reward 时候用
                n_steps=args.n_step,  # (int) The number of steps to run for each environment per update
                # (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
                vf_coef=0.05,  # (float) Value function coefficient for the loss calculation
                ent_coef=0.25,  # (float) Entropy coefficient for the loss calculation
                learning_rate=linear_schedule(1e-4),
                max_grad_norm=args.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log='../logs',
                seed=args.seed
                )
    path = os.path.join(
        father_path,
        'Model',
        args.model_dir,
    )

    model = A2C.load(path, env=env)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=TensorboardCallback()
    )


    print("Save Model to path:", path)
    model.save(path)


#
if __name__ == "__main__":
    # 创建一个参数解析实例
    parser = argparse.ArgumentParser()
    # 添加参数解析
    parser.add_argument("-vertex", nargs='+', type=int, help="Now inserve")
    parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[0, np.inf])    # 如果步输入的话就是np.inf
    # parser.add_argument("-reward_weight", type=float, help=" ", default=1.0)  # 如果步输入的话就是1.0
    parser.add_argument("-model_dir", type=str, help="Save dir")
    parser.add_argument("-reward_scale", nargs='+', type=float, help=" ", default=[1.0, 1.0])
    parser.add_argument("-n_step", type=int, help="n_step", default=5)
    parser.add_argument("-seed", type=int, help="seed", default=94566)
    parser.add_argument("-max_grad_norm", type=float, help="seed", default=10000)
    parser.add_argument("-total_timesteps", type=int, help="seed", default=int(3e7))
    # 开始解析
    args = parser.parse_args()

    train(args)
