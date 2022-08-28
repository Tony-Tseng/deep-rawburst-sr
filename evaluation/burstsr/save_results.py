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

from dataset.burstsr_dataset import get_burstsr_val_set, CanonImage
# from dataset.burstsr_dataset_bip import BurstSRDataset
import torch

import argparse
import importlib
import cv2
import numpy as np
import tqdm
from admin.environment import env_settings


def save_results(setting_name, process=False):
    """ Saves network outputs on the BurstSR validation set. setting_name denotes the name of the experiment
        setting to be used. """
    expr_module = importlib.import_module('evaluation.burstsr.experiments.{}'.format(setting_name))
    expr_func = getattr(expr_module, 'main')
    network_list = expr_func()

    base_results_dir = env_settings().save_data_path

    dataset = get_burstsr_val_set()

    for n in network_list:
        out_dir = '{}/burstsr/{}'.format(base_results_dir, n.get_unique_name())
        if process:
            out_dir = '{}/burstsr/process/{}'.format(base_results_dir, n.get_unique_name())
        os.makedirs(out_dir, exist_ok=True)

        net = n.load_net()
        device = 'cuda:0'
        net.to(device).train(False)

        for idx in tqdm.tqdm(range(len(dataset))):
            data = dataset[idx]
            burst = data['burst'].unsqueeze(0)
            burst_name = data['burst_name']
            meta_info_burst = data['meta_info_burst']

            burst = burst.to(device)

            if n.burst_sz is not None:
                burst = burst[:, :n.burst_sz]

            with torch.no_grad():
                net_pred, _ = net(burst)
            
            if process:
                pred_proc_np = CanonImage.generate_processed_image(net_pred[0].cpu(), meta_info_burst, return_np=True,
                                                                gamma=True,
                                                                smoothstep=True,
                                                                no_white_balance=False,
                                                                external_norm_factor=None)
                pred_proc_np = cv2.cvtColor(pred_proc_np, cv2.COLOR_RGB2BGR)
            else:
                pred_proc_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(
                np.uint16)

            # Save predictions as png
            cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), pred_proc_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saves network outputs on the BurstSR validation set. setting_name '
                                                 'denotes the name of the experiment setting to be used. ')
    parser.add_argument('setting', type=str, help='Name of experiment setting')
    parser.add_argument('--process', action='store_true', default=False)

    args = parser.parse_args()

    save_results(args.setting, args.process)