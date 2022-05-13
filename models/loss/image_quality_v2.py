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

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.loss.msssim as msssim
import models.loss.spatial_color_alignment as sca_utils
import math
import lpips
from torchvision.models import vgg16

class CosineError(nn.Module):
    """ Computes pixel-wise error using the specified metric. Optionally boundary pixels are ignored during error
        calculation """
    def __init__(self, boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.criterion = nn.CosineSimilarity()

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Valid indicates image regions which should be used for loss calculation
        if valid is None:
            err = self.criterion(pred, gt)
        else:
            err = self.criterion(pred, gt, reduction='none')

            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        return err
          
class PixelWiseError(nn.Module):
    """ Computes pixel-wise error using the specified metric. Optionally boundary pixels are ignored during error
        calculation """
    def __init__(self, metric='l1', boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

        if metric == 'l1':
            self.loss_fn = F.l1_loss
        elif metric == 'l2':
            self.loss_fn = F.mse_loss
        elif metric == 'l2_sqrt':
            def l2_sqrt(pred, gt):
                return (((pred - gt) ** 2).sum(dim=-3)).sqrt().mean()
            self.loss_fn = l2_sqrt
        elif metric == 'charbonnier':
            def charbonnier(pred, gt):
                eps = 1e-3
                return ((pred - gt) ** 2 + eps**2).sqrt().mean()
            self.loss_fn = charbonnier
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Valid indicates image regions which should be used for loss calculation
        if valid is None:
            err = self.loss_fn(pred, gt)
        else:
            err = self.loss_fn(pred, gt, reduction='none')

            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        return err


class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = PixelWiseError(metric='l2', boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        if getattr(self, 'max_value', 1.0) is not None:
            psnr = 20 * math.log10(getattr(self, 'max_value', 1.0)) - 10.0 * mse.log10()
        else:
            psnr = 20 * gt.max().log10() - 10.0 * mse.log10()

        if torch.isinf(psnr) or torch.isnan(psnr):
            print('invalid psnr')

        return psnr

    def forward(self, pred, gt, valid=None):
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]

        psnr_all = [p for p in psnr_all if not (torch.isinf(p) or torch.isnan(p))]

        if len(psnr_all) == 0:
            psnr = 0
        else:
            psnr = sum(psnr_all) / len(psnr_all)
        return psnr


class SSIM(nn.Module):
    def __init__(self, boundary_ignore=None, use_for_loss=True):
        super().__init__()
        self.ssim = msssim.SSIM(spatial_out=True)
        self.boundary_ignore = boundary_ignore
        self.use_for_loss = use_for_loss

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        loss = self.ssim(pred, gt)

        if valid is not None:
            valid = valid[..., 5:-5, 5:-5]  # assume window size 11

            eps = 1e-12
            elem_ratio = loss.numel() / valid.numel()
            loss = (loss * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)
        else:
            loss = loss.mean()

        if self.use_for_loss:
            loss = 1.0 - loss
        return loss


class LPIPS(nn.Module):
    def __init__(self, boundary_ignore=None, type='alex', bgr2rgb=False):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.bgr2rgb = bgr2rgb

        if type == 'alex':
            self.loss = lpips.LPIPS(net='alex')
        elif type == 'vgg':
            self.loss = lpips.LPIPS(net='vgg')
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.bgr2rgb:
            pred = pred[..., [2, 1, 0], :, :].contiguous()
            gt = gt[..., [2, 1, 0], :, :].contiguous()

        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        loss = self.loss(pred, gt)

        return loss.mean()


class AlignedL2(nn.Module):
    """ Computes L2 error after performing spatial and color alignment of the input image to GT"""
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sca = sca_utils.SpatialColorAlignment(alignment_net, sr_factor)
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, burst_input):
        pred_warped_m, valid = self.sca(pred, gt, burst_input)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        mse = F.mse_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse

class PerceptualError(nn.Module):
    """_summary_
    Reference: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49

    Args:
        nn (_type_): _description_
    """
    def __init__(self, boundary_ignore=None, resize=True):
        super(PerceptualError, self).__init__()
        self.boundary_ignore = boundary_ignore
        blocks = []
        vgg_model = vgg16(pretrained=True).to("cuda:0")
        blocks.append(vgg_model.features[:4].eval())
        blocks.append(vgg_model.features[4:9].eval())
        blocks.append(vgg_model.features[9:16].eval())
        blocks.append(vgg_model.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).to("cuda:0").view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).to("cuda:0").view(1, 3, 1, 1))


    def forward(self, pred, gt, valid=None, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            input = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            target = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
                
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class SimSiamError(nn.Module):
    def __init__(self, boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        # self.loss_fn = torch.norm
        
    def forward(self, pred, proj, valid=None):
        proj = proj.detach()
        
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            proj = proj[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
                
        if valid is None:
            pred_norm = F.normalize(pred, dim=-1, p=2)
            proj_norm = F.normalize(proj, dim=-1, p=2)
            err = 2 - 2 * (pred_norm * proj_norm).sum(dim=-1).mean()
        else:
            pred_norm = F.normalize(pred, dim=-1, p=2)
            proj_norm = F.normalize(proj, dim=-1, p=2)
            err = 2 - 2 * (pred_norm * proj_norm).sum(dim=-1)

            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        return err   