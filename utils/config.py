import argparse
from omegaconf import OmegaConf
import os
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for MotionTextClassifier")

    # 添加命令行参数，并设置默认值
    parser.add_argument("--config_path", type=str, default="./config.yaml", help="Path to the configuration file")

    return parser.parse_args()

def load_config():
    args = parse_args()

    # 加载 YAML 配置
    config = OmegaConf.load(args.config_path)

    return config