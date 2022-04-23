import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn
import kernel_option

env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)

import admin.settings as ws_settings


def run_training(train_module, train_name, args, cudnn_benchmark=True):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = '{}/{}'.format(train_module, train_name)
    

    expr_module = importlib.import_module('train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'kernel_run')

    expr_func(settings, args)


def main():
    args = kernel_option.args
    
    run_training(args.train_module, args.train_name, args, args.cudnn_benchmark)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
