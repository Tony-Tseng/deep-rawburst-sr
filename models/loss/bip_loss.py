from models.layers.filtering import get_gaussian_kernel
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import numpy as np


class AlignedL1(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        l1 = F.l1_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        l1 = l1 + eps
        elem_ratio = l1.numel() / valid.numel()
        l1 = (l1 * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return l1


class AlignedLPIPS_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    def lpips(self, pred, gt, burst_input):

        #### PSNR
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)
        var1 = 2*pred_warped_m-1
        var2 = 2*gt-1
        LPIPS = self.loss_fn_vgg(var1, var2)
        LPIPS = torch.squeeze(LPIPS)
        
        return LPIPS

    def forward(self, pred, gt, burst_input):
        lpips_all = [self.lpips(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        lpips = sum(lpips_all) / len(lpips_all)
        return lpips


class AlignedPSNR(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value    
        
    def psnr(self, pred, gt, burst_input):
        
        #### PSNR
        mse = self.l2(pred, gt, burst_input) + 1e-12
        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, burst_input):
        pred, gt, burst_input = make_patches(pred, gt, burst_input)
        psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr


class AlignedPred(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        return pred_warped_m, gt, valid


class AlignedL2(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

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


def make_patches(output, labels, burst, patch_size=24):    

    stride = patch_size-(burst.size(-1)%patch_size)
    
    burst1 = burst[0].unfold(2,patch_size,stride).unfold(3,patch_size,stride).contiguous()
    burst1 = burst1.view(14,4,burst1.size(2)*burst1.size(3),patch_size,patch_size).permute(2,0,1,3,4)            
 
    output1 = output.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
    output1 = output1.view(3,output1.size(2)*output1.size(3),patch_size*8,patch_size*8).permute(1,0,2,3)

    labels1 = labels.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
    labels1 = labels1[0].view(3,labels1.size(2)*labels1.size(3),patch_size*8,patch_size*8).permute(1,0,2,3)
    
    return output1, labels1, burst1

def warp(feat, flow, mode='bilinear', padding_mode='zeros'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow im1 --> im2

    input flow must be in format (x, y) at every pixel
    feat: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow (x, y)

    """
    B, C, H, W = feat.size()

    # mesh grid
    rowv, colv = torch.meshgrid([torch.arange(0.5, H + 0.5), torch.arange(0.5, W + 0.5)])
    grid = torch.stack((colv, rowv), dim=0).unsqueeze(0).float().to(feat.device)
    grid = grid + flow

    # scale grid to [-1,1]
    grid_norm_c = 2.0 * grid[:, 0] / W - 1.0
    grid_norm_r = 2.0 * grid[:, 1] / H - 1.0

    grid_norm = torch.stack((grid_norm_c, grid_norm_r), dim=1)

    grid_norm = grid_norm.permute(0, 2, 3, 1)

    output = F.grid_sample(feat, grid_norm, mode=mode, padding_mode=padding_mode, align_corners=True)

    return output


def match_colors(im_ref, im_q, im_test, ksz, gauss_kernel):
    """ Estimates a color transformation matrix between im_ref and im_q. Applies the estimated transformation to
        im_test
    """
    gauss_kernel = gauss_kernel.to(im_ref.device)
    bi = 5

    # Apply Gaussian smoothing
    im_ref_mean = apply_kernel(im_ref, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()
    im_q_mean = apply_kernel(im_q, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()

    im_ref_mean_re = im_ref_mean.view(*im_ref_mean.shape[:2], -1)
    im_q_mean_re = im_q_mean.view(*im_q_mean.shape[:2], -1)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        c = torch.lstsq(ir.t(), iq.t())
        c = c.solution[:3]
        c_mat_all.append(c)

    c_mat = torch.stack(c_mat_all, dim=0)
    im_q_mean_conv = torch.matmul(im_q_mean_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_q_mean_conv = im_q_mean_conv.view(im_q_mean.shape)

    err = ((im_q_mean_conv - im_ref_mean) * 255.0).norm(dim=1)

    thresh = 20

    # If error is larger than a threshold, ignore these pixels
    valid = err < thresh

    pad = (im_q.shape[-1] - valid.shape[-1]) // 2
    pad = [pad, pad, pad, pad]
    valid = F.pad(valid, pad)

    upsample_factor = im_test.shape[-1] / valid.shape[-1]
    valid = F.interpolate(valid.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear', align_corners=True)
    valid = valid > 0.9

    # Apply the transformation to test image
    im_test_re = im_test.view(*im_test.shape[:2], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)

    return im_t_conv, valid

def apply_kernel(im, ksz, gauss_kernel):
    shape = im.shape
    im = im.view(-1, 1, *im.shape[-2:])

    pad = [ksz // 2, ksz // 2, ksz // 2, ksz // 2]
    im = F.pad(im, pad, mode='reflect')
    im_mean = F.conv2d(im, gauss_kernel).view(shape)
    return im_mean