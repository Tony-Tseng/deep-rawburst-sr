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

import data.camera_pipeline as rgb2raw
from actors.base_actor import BaseActor
from models.loss.spatial_color_alignment import SpatialColorAlignment


class SSLRAWSyntheticActor(BaseActor):
    """Actor for training DBSR model on synthetic bursts """
    def __init__(self, net, degrad_net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight
        self.degrad_net = degrad_net

    def __call__(self, data):
        # Run network
        # data['burst'] -> (8, 8, 4, 48, 48)
        # data['ssl_gt'] -> (8, 3, 96, 96)
        upper_degrade = self.degrad_net(data['burst'])
        upper_pred, upper_aux_dict = self.net(upper_degrade)
        
        lower_sr, lower_aux_dict = self.net(data['burst'])
        lower_pred = self.degrad_net(lower_sr)
        
        upper_raw_pred = rgb2raw.mosaic(upper_pred)
        lower_raw_pred = rgb2raw.mosaic(lower_pred)

        # Compute loss
        aux_loss = self.objective['ssl'](upper_raw_pred, data['burst'][:, 0, ...])
        deg_loss = self.objective['ssl'](lower_raw_pred, data['burst'][:, 0, ...])

        if 'psnr' in self.objective.keys():
            psnr = self.objective['psnr'](lower_sr.clone().detach(), data['frame_gt'])

        total_loss = deg_loss + aux_loss

        stats = {'Loss/total': total_loss.item(),
                 'Loss/deg': deg_loss.item(),
                 'Loss/aux': aux_loss.item()
                 }

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return total_loss, stats
    
class SSLSyntheticActor(BaseActor):
    """Actor for training DBSR model on synthetic bursts """
    def __init__(self, net, degrad_net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight
        self.degrad_net = degrad_net

    def __call__(self, data):
        # Run network
        # data['burst'] -> (8, 8, 4, 48, 48)
        # data['ssl_gt'] -> (8, 3, 96, 96)
        upper_degrade = self.degrad_net(data['burst'])
        upper_pred, upper_aux_dict = self.net(upper_degrade)
        
        lower_sr, lower_aux_dict = self.net(data['burst'])
        lower_pred = self.degrad_net(lower_sr)

        # Compute loss
        aux_loss = self.objective['ssl'](upper_pred, data['ssl_gt'])
        deg_loss = self.objective['ssl'](lower_pred, data['ssl_gt'])

        if 'psnr' in self.objective.keys():
            psnr = self.objective['psnr'](lower_sr.clone().detach(), data['frame_gt'])

        total_loss = deg_loss + aux_loss

        stats = {'Loss/total': total_loss.item(),
                #  'Loss/rgb': loss_rgb.item(),
                #  'Loss/raw/rgb': loss_rgb_raw.item(),
                 'Loss/deg': deg_loss.item(),
                 'Loss/aux': aux_loss.item()
                 }

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return total_loss, stats
    

class SSLkernelSyntheticActor(BaseActor):
    """Actor for training DBSR model on synthetic bursts """
    def __init__(self, net, degrad_net, kernel_net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight
        self.degrad_net = degrad_net
        self.kernel_net = kernel_net

    def __call__(self, data):
        # Run network
        # data['burst'] -> (8, 8, 4, 48, 48)
        # data['ssl_gt'] -> (8, 3, 96, 96)
        kernel = self.kernel_net(data['base_rgb'].float())
        
        upper_degrade = self.degrad_net(data['burst'], kernel)
        upper_pred, upper_aux_dict = self.net(upper_degrade)
        
        lower_sr, lower_aux_dict = self.net(data['burst'])
        lower_pred = self.degrad_net(lower_sr, kernel)

        # Compute loss
        aux_loss = self.objective['ssl'](upper_pred, data['ssl_gt'])
        deg_loss = self.objective['ssl'](lower_pred, data['ssl_gt'])

        if 'psnr' in self.objective.keys():
            psnr = self.objective['psnr'](lower_sr.clone().detach(), data['frame_gt'])

        total_loss = deg_loss + aux_loss

        stats = {'Loss/total': total_loss.item(),
                #  'Loss/rgb': loss_rgb.item(),
                #  'Loss/raw/rgb': loss_rgb_raw.item(),
                 'Loss/deg': deg_loss.item(),
                 'Loss/aux': aux_loss.item()
                 }

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return total_loss, stats
    
    
class SSL1waySyntheticActor(BaseActor):
    """Actor for training DBSR model on synthetic bursts """
    def __init__(self, net, degrad_net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight
        self.degrad_net = degrad_net

    def __call__(self, data):
        # Run network
        # data['burst'] -> (8, 8, 4, 48, 48)
        # data['ssl_gt'] -> (8, 3, 96, 96)
        sr, lower_aux_dict = self.net(data['burst'])
        pred = self.degrad_net(sr)

        # Compute loss
        aux_loss = self.objective['ssl'](pred, data['ssl_gt'])

        if 'psnr' in self.objective.keys():
            psnr = self.objective['psnr'](sr.clone().detach(), data['frame_gt'])

        total_loss = aux_loss

        stats = {'Loss/total': total_loss.item(),
                #  'Loss/rgb': loss_rgb.item(),
                #  'Loss/raw/rgb': loss_rgb_raw.item(),
                 'Loss/aux': aux_loss.item()
                 }

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return total_loss, stats

class BYOLSyntheticActor(BaseActor):
    """Actor for training DBSR model on synthetic bursts """
    def __init__(self, net, project_net, predict_net, degrad_net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight
        self.project_net = project_net
        self.predict_net = predict_net
        self.degrad_net = degrad_net
    
    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.project_net.to(device)
        self.predict_net.to(device)

    def __call__(self, data):
        # Run network
        # data['burst'] -> (8, 8, 4, 48, 48)
        # data['ssl_gt'] -> (8, 3, 96, 96)
        # upper_degrade = self.predict_net(data['burst'])
        upper_degrade = self.degrad_net(data['burst'])
        upper_latent, upper_aux_dict = self.net(upper_degrade)
        
        lower_sr, lower_aux_dict = self.net(data['burst'])
        lower_latent = self.degrad_net(lower_sr)

        upper_proj = self.project_net(upper_latent)
        lower_proj = self.project_net(lower_latent)
        
        upper_pred = self.predict_net(upper_proj)
        
        # cos = self.objective['cosine'](upper_proj, lower_proj).mean()
        # print(cos)
        
        # Compute loss
        ssl_loss = self.objective['ssl'](upper_pred, lower_proj)

        if 'psnr' in self.objective.keys():
            psnr = self.objective['psnr'](lower_sr.clone().detach(), data['frame_gt'])

        total_loss = ssl_loss

        stats = {'Loss/total': total_loss.item(),
                #  'Loss/rgb': loss_rgb.item(),
                #  'Loss/raw/rgb': loss_rgb_raw.item(),
                 'Loss/ssl': ssl_loss.item()
                 }

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return total_loss, stats


class DBSRRealkernelSSLActor(BaseActor):
    def __init__(self, net, degrad_net, kernel_net, objective, alignment_net, loss_weight=None, sr_factor=4):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}

        self.sca = SpatialColorAlignment(alignment_net.eval(), sr_factor=sr_factor)
        self.loss_weight = loss_weight
        self.degrad_net = degrad_net
        self.kernel_net = kernel_net

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.sca.to(device)
        # self.kernel_net.to(device)

    def __call__(self, data):
        # Run network
        gt = data['frame_gt']
        burst = data['burst']
        # pred, aux_dict = self.net(burst)

        # # Perform spatial and color alignment of the prediction
        # pred_warped_m, valid = self.sca(pred, gt, burst)

        # # Compute loss
        # loss_rgb_raw = self.objective['rgb'](pred_warped_m, gt, valid=valid)

        kernel = self.kernel_net(data['base_rgb'].float())
        
        upper_degrade = self.degrad_net(burst, kernel)
        upper_pred, upper_aux_dict = self.net(upper_degrade)
        
        lower_sr, lower_aux_dict = self.net(burst)
        lower_pred = self.degrad_net(lower_sr, kernel)

        # Compute loss
        aux_loss = self.objective['ssl'](upper_pred, data['ssl_gt'])
        deg_loss = self.objective['ssl'](lower_pred, data['ssl_gt'])
    
        pred_warped_m, valid = self.sca(lower_sr, gt, burst)
        if 'psnr' in self.objective.keys():
            # detach, otherwise there is memory leak
            psnr = self.objective['psnr'](pred_warped_m.clone().detach(), gt, valid=valid)

        loss = deg_loss + aux_loss

        stats = {'Loss/total': loss.item(),
                 'Loss/deg': deg_loss.item(),
                 'Loss/aux': aux_loss.item()
                }

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return loss, stats
    

class DBSRRealWorldSSLActor(BaseActor):
    """Actor for training DBSR model on real-world bursts from BurstSR dataset"""
    def __init__(self, net, degrad_net, objective, alignment_net, loss_weight=None, sr_factor=4):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}

        self.sca = SpatialColorAlignment(alignment_net.eval(), sr_factor=sr_factor)
        self.loss_weight = loss_weight
        self.degrad_net = degrad_net

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
        # pred, aux_dict = self.net(burst)

        # # Perform spatial and color alignment of the prediction
        # pred_warped_m, valid = self.sca(pred, gt, burst)

        # # Compute loss
        # loss_rgb_raw = self.objective['rgb'](pred_warped_m, gt, valid=valid)

        # loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw
        
        upper_degrade = self.degrad_net(burst)
        upper_pred, upper_aux_dict = self.net(upper_degrade)
        
        lower_sr, lower_aux_dict = self.net(burst)
        lower_pred = self.degrad_net(lower_sr)

        # Compute loss
        aux_loss = self.objective['ssl'](upper_pred, data['ssl_gt'])
        deg_loss = self.objective['ssl'](lower_pred, data['ssl_gt'])
    
        pred_warped_m, valid = self.sca(lower_sr, gt, burst)
        if 'psnr' in self.objective.keys():
            # detach, otherwise there is memory leak
            psnr = self.objective['psnr'](pred_warped_m.clone().detach(), gt, valid=valid)

        loss = deg_loss + aux_loss

        stats = {'Loss/total': loss.item(),
                 'Loss/deg': deg_loss.item(),
                 'Loss/aux': aux_loss.item()
                }

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return loss, stats