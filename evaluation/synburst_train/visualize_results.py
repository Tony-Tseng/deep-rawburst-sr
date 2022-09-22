# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from utils.opencv_plotting import BurstSRVis
import torch
from models.loss.metrics import PSNR
import cv2
import numpy as np
import argparse
import importlib
from data_processing.camera_pipeline import process_linear_image_rgb
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.loader import DataLoader
from options.base_option import Environment_Setting


def visualize_results(setting_name):
    """ Visualize the results on the SyntheticBurst validation set. setting_name denotes
        the name of the experiment setting, which contains the list of methods for which to visualize results.
    """

    expr_module = importlib.import_module(f'evaluation.synburst_train.{setting_name}')
    expr_func = getattr(expr_module, 'main')
    network_list = expr_func()

    settings = Environment_Setting()
    base_results_dir = settings.save_data_path
    metric = PSNR(boundary_ignore=None)
    vis = BurstSRVis(boundary_ignore=40, metric=metric)

    zurich_raw2rgb_val = ZurichRAW2RGB(settings.zurichraw2rgb_dir,
                                       split='test')
    test_dataset = SyntheticBurst(zurich_raw2rgb_val,
                                  burst_size=14,
                                  crop_sz=384,
                                  add_noise=False)

    dataset = DataLoader('val', test_dataset, batch_size=1)

    for idx, data in enumerate(dataset, 1):
        burst, gt, meta_info = data["burst"], data["frame_gt"], data["meta_info"]
        burst_name = f'{idx:04d}'
        gt = gt.squeeze()
        meta_info["cam2rgb"] = meta_info["cam2rgb"].squeeze()

        pred_all = []
        titles_all = []
        for n in network_list:
            pred_path = f"{base_results_dir}/synburst_train/{n.get_unique_name()}/{burst_name}.png"
            pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
            pred = torch.from_numpy(pred.astype(np.float32) / 2 ** 14).permute(2, 0, 1)
            pred_all.append(pred)
            titles_all.append(n.get_display_name())

        gt = process_linear_image_rgb(gt, meta_info, return_np=True)
        pred_all = [process_linear_image_rgb(p, meta_info, return_np=True) for p in pred_all]
        data = [{'images': [gt, ] + pred_all,
                 'titles': [burst_name, ] + titles_all}]
        cmd = vis.plot(data)

        if cmd == 'stop':
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the results on the SyntheticBurst validation set. '
                                                 'setting_name denotes the name of the experiment setting, which '
                                                 'contains the list of methods for which to visualize results.')
    parser.add_argument('setting', type=str, help='Name of experiment setting')

    args = parser.parse_args()

    print('Press \'n\' to show next image. \n'
          'Press \'q\' to quit. \n'
          'Zoom on a particular region by drawing a box around it (click on the two corner points). \n'
          'In the zoomed pane (last row), you can click on an image an drag around. \n'
          'Using \'w\' and \'s\' keys, you can navigate between the two panes (normal pane and zoom pane) \n'
          'Using the \'space\' key, you can toggle between showing all the images and showing only a single image. \n' 
          'In the single image mode, you can navigate between different images using the \'a\' and \'d\' keys. \n')

    visualize_results(args.setting)
