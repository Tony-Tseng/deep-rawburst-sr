import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
import models.layers.blocks as blocks
from admin.model_constructor import model_constructor
from models.layers.upsampling import PixShuffleUpsampler
from deformable_attention import DeformableAttention
from models.DCN.dcnsr import EBFA


class MergeBlockDiff(nn.Module):
    def __init__(self, input_dim, project_dim, num_weight_predictor_res=1, 
                 use_bn=False, activation='relu'):
        super().__init__()
        
        weight_predictor = []
        weight_predictor.append(blocks.conv_block(input_dim, 2 * project_dim, 3,
                                                  stride=1, padding=1, batch_norm=use_bn, activation=activation))

        for _ in range(num_weight_predictor_res):
            weight_predictor.append(blocks.ResBlock(2 * project_dim, 2 * project_dim, stride=1,
                                                    batch_norm=use_bn, activation=activation))

        weight_predictor.append(blocks.conv_block(2 * project_dim, input_dim, 3, stride=1, padding=1,
                                                  batch_norm=use_bn,
                                                  activation='none'))

        self.weight_predictor = nn.Sequential(*weight_predictor)
    
    def forward(self, feat):
        base_feat_proj = feat.mean(dim=0, keepdim=True)
        feat_diff = feat - base_feat_proj
        
        weight = self.weight_predictor(feat)
        # To-do: Modify dim=1 to dim=0
        weights_norm = F.softmax(weight, dim=0)
        fused_feat = (feat * weights_norm).sum(dim=0)
        
        return fused_feat.unsqueeze(0)

        
class ResPixShuffleConv(nn.Module):
    """ Decoder employing sub-pixel convolution for upsampling. Passes the input feature map first through a residual
        network. The output features are upsampled using sub-pixel convolution and passed through additional
        residual blocks. A final conv layer generates the output RGB image. """
    def __init__(self, input_dim, init_conv_dim, num_pre_res_blocks, post_conv_dim,
                 num_post_res_blocks,
                 use_bn=False, activation='relu',
                 upsample_factor=2, icnrinit=False, gauss_blur_sd=None, gauss_ksz=3):
        super().__init__()
        self.gauss_ksz = gauss_ksz
        self.init_layer = blocks.conv_block(input_dim, init_conv_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                            activation=activation)

        d_in = init_conv_dim
        pre_res_layers = []

        for _ in range(num_pre_res_blocks):
            pre_res_layers.append(blocks.ResBlock(d_in, d_in, stride=1, batch_norm=use_bn, activation=activation))

        self.pre_res_layers = nn.Sequential(*pre_res_layers)

        self.upsample_layer = PixShuffleUpsampler(d_in, post_conv_dim, upsample_factor=upsample_factor,
                                                  use_bn=use_bn, activation=activation, icnrinit=icnrinit,
                                                  gauss_blur_sd=gauss_blur_sd, gauss_ksz=gauss_ksz)

        post_res_layers = []
        for _ in range(num_post_res_blocks):
            post_res_layers.append(blocks.ResBlock(post_conv_dim, post_conv_dim, stride=1, batch_norm=use_bn,
                                                   activation=activation))

        self.post_res_layers = nn.Sequential(*post_res_layers)

        self.predictor = blocks.conv_block(post_conv_dim, 3, 1, stride=1, padding=0, batch_norm=False)

    def forward(self, feat):
        assert feat.dim() == 4
        out = self.pre_res_layers(self.init_layer(feat))
        out = self.upsample_layer(out)

        pred = self.predictor(self.post_res_layers(out))
        return pred


class DATSRNet(nn.Module):
    def __init__(self, alignment, attention, fusion, decoder):
        super().__init__()

        self.alignment = alignment
        self.attention = attention
        self.fusion = fusion
        self.decoder = decoder

    def forward(self, im):
        out_attn = self.attention(im[0]) # [1 14 4 48 48]
        out_enc = self.alignment(out_attn.unsqueeze(0))
        out_fus = self.fusion(out_enc)
        out_dec = self.decoder(out_fus)

        return out_dec, out_fus


@model_constructor
def datsrnet(alignment_init_dim, reduction, alignment_out_dim, dec_init_conv_dim, 
             dec_num_pre_res_blocks, dec_post_conv_dim, dec_num_post_res_blocks, 
             upsample_factor=2, activation='relu', icnrinit=False, 
             gauss_blur_sd=None, gauss_ksz=3, 
             ):

    ebfa = EBFA(num_features=alignment_init_dim, reduction=reduction)

    attention = DeformableAttention(
        dim = 4,                   # feature dimensions
        dim_head = 64,              # dimension per head
        heads = 8,                  # attention heads
        dropout = 0.,               # dropout
        downsample_factor = 4,      # downsample factor (r in paper)
        offset_scale = 4,           # scale of offset, maximum offset
        offset_groups = 1,          # number of offset groups, should be multiple of heads
        offset_kernel_size = 6,     # offset kernel size
    )
    fusion = MergeBlockDiff(input_dim=64, project_dim=64)
    decoder = ResPixShuffleConv(alignment_out_dim, dec_init_conv_dim, dec_num_pre_res_blocks,
                                dec_post_conv_dim, dec_num_post_res_blocks,
                                upsample_factor=upsample_factor, activation=activation,
                                gauss_blur_sd=gauss_blur_sd, icnrinit=icnrinit,
                                gauss_ksz=gauss_ksz)

    net = DATSRNet(alignment=ebfa, attention=attention, fusion=fusion, decoder=decoder)

    return net


# @model_constructor
# def datsrnet(alignment_out_dim, dec_init_conv_dim, dec_num_pre_res_blocks,
#              dec_post_conv_dim, dec_num_post_res_blocks, upsample_factor=2,
#              activation='relu', icnrinit=False, gauss_blur_sd=None, gauss_ksz=3, 
#              ):
    
#     patch_size: 4
#     num_classes: 1000
#     expansion: 4
#     dim_stem: 96
#     dims: [96, 192, 384, 768]
#     depths: [2, 2, 18, 2]
#     stage_spec: [[L, S], [L, S], [L, D, L, D, L, D, L, D, L, D, L, D, L, D, L, D, L, D], [L, D]]
#     heads: [3, 6, 12, 24]
#     window_sizes: [7, 7, 7, 7] 
#     groups: [-1, -1, 3, 6]
#     sr_ratios: [-1, -1, -1, -1]
#     use_dwc_mlps: [False, False, False, False]
#     use_conv_patches: False
#     drop_rate: 0.0
#     attn_drop_rate: 0.0
#     drop_path_rate: 0.3

#     fmap_size = 224
#     use_pes = [False, False, True, True]
#     dwc_pes = [False, False, False, False]
#     strides = [-1, -1, 1, 1]
#     offset_range_factor = [-1, -1, 2, 2]
#     no_offs = [False, False, False, False]
#     fixed_pe = [False, False, False, False]

#     alignment = DAttentionBaseline(fmap_size, fmap_size, heads, 
#                 hc, n_groups, attn_drop, proj_drop, 
#                 stride, offset_range_factor, use_pe, dwc_pe, 
#                 no_off, fixed_pe, stage_idx)
#     fusion = MergeBlockDiff(input_dim=64, project_dim=64)
#     decoder = ResPixShuffleConv(alignment_out_dim, dec_init_conv_dim, dec_num_pre_res_blocks,
#                                 dec_post_conv_dim, dec_num_post_res_blocks,
#                                 upsample_factor=upsample_factor, activation=activation,
#                                 gauss_blur_sd=gauss_blur_sd, icnrinit=icnrinit,
#                                 gauss_ksz=gauss_ksz)

#     net = DATSRNet(alignment=, fusion=fusion, decoder=decoder)

#     return net