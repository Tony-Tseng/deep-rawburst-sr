import torch
import torch.nn as nn
from turtle import forward
from torchvision.ops import DeformConv2d
import models.layers.blocks as blocks
from models.layers.upsampling import PixShuffleUpsampler
from admin.model_constructor import model_constructor
import torch.nn.functional as F
import models.DCN.upsample.swin_util as swin_util

from timm.models.layers import to_2tuple

##############################################################################################
######################### Residual Global Context Attention Block ##########################################
##############################################################################################

class RGCAB(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RGCAB, self).__init__()
        self.module = [RGCA(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)

class RGCA(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), groups =1):

        super(RGCA, self).__init__()

        self.n_feat = n_feat
        self.groups = groups
        self.reduction = reduction

        modules_body = [nn.Conv2d(n_feat, n_feat, 3,1,1 , bias=bias, groups=groups), act, nn.Conv2d(n_feat, n_feat, 3,1,1 , bias=bias, groups=groups)]
        self.body   = nn.Sequential(*modules_body)

        self.gcnet = nn.Sequential(GCA(n_feat, n_feat))
        self.conv1x1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.gcnet(res)
        res = self.conv1x1(res)
        res += x
        return res


######################### Global Context Attention ##########################################

class GCA(nn.Module):
    def __init__(self, inplanes, planes, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), bias=False):
        super(GCA, self).__init__()

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
            act,
            nn.Conv2d(planes, inplanes, kernel_size=1, bias=bias)
        )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

class EBFA(nn.Module):
    """ Deformable Convolution Super-Resolution """
    def __init__(self, num_features=64, reduction=8, bias=False):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(4, num_features, kernel_size=3, padding=1, bias=bias))

        ####### Edge Boosting Feature Alignment
        
        ## Feature Processing Module
        self.encoder = nn.Sequential(*[RGCAB(num_features, 3, reduction) for _ in range(3)])

        ## Burst Feature Alignment

        # Offset Setting
        kernel_size = 3
        deform_groups = 8
        out_channels = deform_groups * 3 * kernel_size**2
        
        self.bottleneck = nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=1, bias=bias)

        # Offset Conv
        self.offset_conv1 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        self.offset_conv2 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        self.offset_conv3 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        
        # Deform Conv
        self.deform1 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform2 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform3 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        
        ## Refined Aligned Feature
        self.feat_ext1 = nn.Sequential(*[RGCAB(num_features, 3, reduction) for _ in range(3)])
        self.cor_conv1 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=bias))

    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return offset, mask
    
    
    def def_alignment(self, burst_feat):
        
        b, f, H, W = burst_feat.size()
        ref = burst_feat[0].unsqueeze(0)
        ref = torch.repeat_interleave(ref, b, dim=0)

        feat = self.bottleneck(torch.cat([ref, burst_feat], dim=1))

        offset1, mask1 = self.offset_gen(self.offset_conv1(feat))
        feat = self.deform1(feat, offset1, mask1)
        
        offset2, mask2 = self.offset_gen(self.offset_conv2(feat))
        feat = self.deform2(feat, offset2, mask2)
        
        offset3, mask3 = self.offset_gen(self.offset_conv3(feat))
        aligned_feat = self.deform3(burst_feat, offset3, mask3)     
       
        return aligned_feat
    
    def forward(self, burst):
         ################### 
        # Input: (B, 4, H/2, W/2)    
        # Output: (1, 3, 4H, 4W)
        ###################
        
        burst = burst[0]
        burst_feat = self.conv1(burst)          # (B, num_features, H/2, W/2)

        ##################################################
        ####### Edge Boosting Feature Alignment #################
        ##################################################

        base_frame_feat = burst_feat[0].unsqueeze(0)
        burst_feat = self.encoder(burst_feat)               
        
        ## Burst Feature Alignment
        burst_feat = self.def_alignment(burst_feat)

        ## Refined Aligned Feature
        burst_feat = self.feat_ext1(burst_feat)                
        Residual = burst_feat - base_frame_feat
        Residual = self.cor_conv1(Residual)
        burst_feat = Residual + burst_feat          # (B, num_features, H/2, W/2)
        
        return burst_feat

class MergeBlockUNetDiff(nn.Module):
    def __init__(self, input_dim, project_dim, 
                 num_weight_predictor_res=1, use_bn=False, bias=False, 
                 activation='relu'):
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
        self.up = nn.Conv2d(input_dim, 240, 3, 1, 1)
    
    def forward(self, feat):
        weight = self.weight_predictor(feat)
        weights_norm = F.softmax(weight, dim=0)
        fused_feat = (feat * weights_norm).sum(dim=0)

        fused_feat = self.up(fused_feat)
        
        return fused_feat.unsqueeze(0)


class BSRT_ResPixUpSample(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, num_feat, embed_dim):
        super().__init__()

        self.center = 0
        # num_in_ch = 4
        # num_out_ch = 3
        # num_feat = 64
        # embed_dim = 96
        img_size = 64
        patch_size = 1
        drop_rate = 0.
        norm_layer = nn.LayerNorm

        depths = [6, 6, 6, 6]
        self.num_layers = len(depths)
        num_heads=[6, 6, 6, 6]
        window_size = 8

        self.skip_pixel_shuffle = nn.PixelShuffle(2)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.upconv1 = nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        # self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)

        self.norm = norm_layer(embed_dim)
        # split image into non-overlapping patches
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_embed = swin_util.PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer)
        patches_resolution = self.patch_embed.patches_resolution

        self.layers = nn.ModuleList()
        for i_layer in range(1, self.num_layers):
            layer = swin_util.RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size
                         )
            self.layers.append(layer)

        # merge non-overlapping patches into image
        self.patch_unembed = swin_util.PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer)


    def forward(self, fused_feat):
        fused_feat = self.skip_pixel_shuffle(fused_feat)
        fused_feat = self.lrelu(self.conv_after_body(self.forward_features(fused_feat))) + fused_feat

        fused_feat = self.lrelu(self.pixel_shuffle(self.upconv1(fused_feat)))
        fused_feat = self.lrelu(self.pixel_shuffle(self.upconv2(fused_feat)))
        fused_feat = self.lrelu(self.HRconv(fused_feat))
        fused_feat = self.conv_last(fused_feat)

        return fused_feat
    
    def forward_features(self, x):
        x_size = (x.shape[-2], x.shape[-1])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, x_size)
            if torch.any(torch.isinf(x)) or torch.any(torch.isnan(x)):
                print('layer: ', idx)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x


class DCNSRNet(nn.Module):
    """ DCN model"""
    def __init__(self, alignment, fusion, decoder):
        super().__init__()

        self.alignment = alignment  # Encodes input images and performs alignment
        self.fusion = fusion
        self.decoder = decoder      # Decodes the merged embeddings to generate HR RGB image

    def forward(self, im):
        out_enc = self.alignment(im)
        out_fus = self.fusion(out_enc)
        out_dec = self.decoder(out_fus)

        return out_dec, out_fus


@model_constructor
def dcnsrnet_unet_mergediff(alignment_init_dim, reduction):
    
    ebfa = EBFA(num_features=alignment_init_dim, reduction=reduction)
    fusion = MergeBlockUNetDiff(input_dim=64, project_dim=64)
    decoder = BSRT_ResPixUpSample(num_in_ch=4, num_out_ch=3, num_feat=64, embed_dim=60)
    
    net = DCNSRNet(alignment=ebfa, fusion=fusion, decoder = decoder)
    
    return net