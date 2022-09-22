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

import torch, random

import argparse
import importlib
import numpy as np
import cv2
import tqdm
from evaluation.common_utils.synthetic_burst_train_set_for_vis import SyntheticBurst
from admin.environment import env_settings
import dataset as datasets
from data.loader import DataLoader
from data.postprocessing_functions import process_linear_image_rgb


def save_results(setting_name, process=False):
    """ Saves network outputs on the SyntheticBurst validation set. setting_name denotes the name of the experiment
        setting to be used. """
    random.seed(10)
    expr_module = importlib.import_module(
        'evaluation.synburst_train.{}'.format(setting_name))
    expr_func = getattr(expr_module, 'main')
    network_list = expr_func()

    base_results_dir = env_settings().save_data_path
    zurich_raw2rgb_val = datasets.ZurichRAW2RGB(split='train')
    test_dataset = SyntheticBurst(zurich_raw2rgb_val,
                                  burst_size=14,
                                  crop_sz=384,
                                  add_noise=True)

    dataset = DataLoader('val', test_dataset, batch_size=1)

    for n in network_list:
        net = n.load_net()
        device = 'cuda'
        net.to(device).train(False)

        out_dir = '{}/synburst_train/{}'.format(base_results_dir,
                                                n.get_unique_name())
        if process:
            out_dir = '{}/synburst_train_processed/{}'.format(
                base_results_dir, n.get_unique_name())
        os.makedirs(out_dir, exist_ok=True)

        for idx, data in enumerate(dataset, 1):
            if idx > 50:
                break
            burst, meta_info = data["burst"], data["meta_info"]
            burst = burst.to(device)
            meta_info["cam2rgb"] = meta_info["cam2rgb"].squeeze()

            with torch.no_grad():
                net_pred, _ = net(burst)

                if process:
                    net_pred_np = process_linear_image_rgb(
                        net_pred.squeeze(0).cpu(), meta_info, return_np=True)
                    net_pred_np = cv2.cvtColor(net_pred_np, cv2.COLOR_RGB2BGR)
                else:
                    # Normalize to 0  2^14 range and convert to numpy array
                    net_pred_np = (
                        net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) *
                        2**14).cpu().numpy().astype(np.uint16)

                # Save predictions as png
                cv2.imwrite(f'{out_dir}/{idx:04d}.png', net_pred_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Saves network outputs on the SyntheticBurst validation set. '
        'setting_name denotes the name of the experiment setting to be used.')
    parser.add_argument('setting', type=str, help='Name of experiment setting')
    parser.add_argument('--process', action='store_true', default=False)

    args = parser.parse_args()

    save_results(args.setting, args.process)