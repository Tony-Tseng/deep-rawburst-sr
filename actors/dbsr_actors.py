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

from actors.base_actor import BaseActor
from models.loss.spatial_color_alignment import SpatialColorAlignment
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_flow(ground, pred):
    ground_flow = ground[:,1:,:,::2, ::2].cpu().numpy()
    predict_flow = pred.cpu().numpy()
    
    ground_mean_flow = np.mean(ground_flow, axis=(3,4))
    predict_mean_flow = np.mean(predict_flow, axis=(3,4))
    
    num = np.random.randint(1000000)
    plt.figure(num)
    # for i in range(ground_mean_flow.shape[1]):
    plt.arrow(0, 0, ground_mean_flow[0, 0, 1], ground_mean_flow[0, 0, 1], color='red', width = 0.01, head_width = 0.1)
    plt.arrow(0, 0, predict_mean_flow[0, 0, 1], predict_mean_flow[0, 0, 1], color='blue', width = 0.01, head_width = 0.1)
    plt.legend(['Ground optical', 'Predict optical'])
    plt.title('Predict & Ground Optical Flow')
    plt.savefig(f"sample_plot/flow/flow_{num}.png")

    
def calc_flow(ground, pred):
    ground_flow = ground[:,1:,:,::2, ::2].cpu().numpy()
    predict_flow = pred.cpu().numpy()
    
    ground_mean_flow = np.mean(ground_flow, axis=(3,4))
    predict_mean_flow = np.mean(predict_flow, axis=(3,4))
    
    diff_flow = ground_mean_flow - predict_mean_flow
    norm_flow = np.linalg.norm(diff_flow, axis=2)

    return norm_flow.reshape(-1, 1)
    
    
class DBSRSyntheticActor(BaseActor):
    """Actor for training DBSR model on synthetic bursts """
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data, flow_array):
        # Run network
        pred, aux_dict = self.net(data['burst'])
        
        plot = True
        if plot:
            plot_flow(data["flow"], -aux_dict["offsets"])
            # flow_array = np.vstack((flow_array, calc_flow(data["flow"], -aux_dict["offsets"])))

        # Compute loss
        loss_rgb_raw = self.objective['rgb'](pred, data['frame_gt'])
        loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw

        if 'psnr' in self.objective.keys():
            psnr = self.objective['psnr'](pred.clone().detach(), data['frame_gt'])

        loss = loss_rgb

        stats = {'Loss/total': loss.item(),
                 'Loss/rgb': loss_rgb.item(),
                 'Loss/raw/rgb': loss_rgb_raw.item()}

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return loss, stats, flow_array

# class DBSRSyntheticActor(BaseActor):
#     """Actor for training DBSR model on synthetic bursts """
#     def __init__(self, net, objective, loss_weight=None):
#         super().__init__(net, objective)
#         if loss_weight is None:
#             loss_weight = {'rgb': 1.0}
#         self.loss_weight = loss_weight

#     def __call__(self, data):
#         # Run network
#         pred, aux_dict = self.net(data['burst'])

#         # Compute loss
#         loss_rgb_raw = self.objective['rgb'](pred, data['frame_gt'])
#         loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw

#         if 'psnr' in self.objective.keys():
#             psnr = self.objective['psnr'](pred.clone().detach(), data['frame_gt'])

#         loss = loss_rgb

#         stats = {'Loss/total': loss.item(),
#                  'Loss/rgb': loss_rgb.item(),
#                  'Loss/raw/rgb': loss_rgb_raw.item()}

#         if 'psnr' in self.objective.keys():
#             stats['Stat/psnr'] = psnr.item()

#         return loss, stats


class DBSRRealWorldActor(BaseActor):
    """Actor for training DBSR model on real-world bursts from BurstSR dataset"""
    def __init__(self, net, objective, alignment_net, loss_weight=None, sr_factor=4):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}

        self.sca = SpatialColorAlignment(alignment_net.eval(), sr_factor=sr_factor)
        self.loss_weight = loss_weight

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.sca.to(device)

    def __call__(self, data):
        # Run network
        gt = data['frame_gt']
        burst = data['burst']
        pred, aux_dict = self.net(burst)

        # Perform spatial and color alignment of the prediction
        pred_warped_m, valid = self.sca(pred, gt, burst)

        # Compute loss
        loss_rgb_raw = self.objective['rgb'](pred_warped_m, gt, valid=valid)

        loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw

        if 'psnr' in self.objective.keys():
            # detach, otherwise there is memory leak
            psnr = self.objective['psnr'](pred_warped_m.clone().detach(), gt, valid=valid)

        loss = loss_rgb

        stats = {'Loss/total': loss.item(),
                 'Loss/rgb': loss_rgb.item(),
                 'Loss/raw/rgb': loss_rgb_raw.item()}

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return loss, stats
