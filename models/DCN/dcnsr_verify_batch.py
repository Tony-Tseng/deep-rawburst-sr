import torch
import torch.nn as nn
from turtle import forward
from torchvision.ops import DeformConv2d
import models.layers.blocks as blocks
from models.layers.upsampling import PixShuffleUpsampler
from admin.model_constructor import model_constructor
import torch.nn.functional as F

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
        
        self.num_features = num_features
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
        self.offset_conv4 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        
        # Deform Conv
        self.deform1 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform2 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform3 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform4 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        
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
        feat = self.deform3(burst_feat, offset3, mask3)
        
        offset4, mask4 = self.offset_gen(self.offset_conv4(feat))
        aligned_feat = self.deform4(feat, offset4, mask4)        
       
        return aligned_feat
    
    def forward(self, burst):
         ################### 
        # Input: (B, 4, H/2, W/2)    
        # Output: (1, 3, 4H, 4W)
        ###################
        B, N, C, H, W = burst.shape
        # base_frame = burst[:, 0, ...]

        # burst = burst[0]
        burst = burst.view(-1, C, H, W)         # B * N, C, H/2, W/2
        burst_feat = self.conv1(burst)          # (B, num_features, H/2, W/2)

        # burst_feat = burst_feat.view(B, N, self.num_features, H, W)
        ##################################################
        ####### Edge Boosting Feature Alignment #################
        ##################################################
        
        base_frame_feat = burst_feat[::N, ...].unsqueeze(1)
        base_frame_feat = base_frame_feat.repeat(1, N, 1, 1, 1)
        base_frame_feat = base_frame_feat.view(-1, self.num_features, H, W)
        # burst_feat = burst_feat.view(-1, self.num_features, H, W)

        # base_frame_feat = burst_feat[0].unsqueeze(0)
        burst_feat = self.encoder(burst_feat)

        ## Burst Feature Alignment
        burst_feat = self.def_alignment(burst_feat)

        ## Refined Aligned Feature
        burst_feat = self.feat_ext1(burst_feat)                
        Residual = burst_feat - base_frame_feat
        Residual = self.cor_conv1(Residual)
        burst_feat = Residual + burst_feat          # (B, num_features, H/2, W/2)
        
        return burst_feat #.view(B, N, C, H, W)

##############################################################################################
######################### Multi-scale Feature Extractor ##########################################
##############################################################################################

class UpSample(nn.Module):

    def __init__(self, in_channels, chan_factor, bias=False):
        super(UpSample, self).__init__()

        self.up = nn.Sequential(nn.Conv2d(in_channels, int(in_channels/chan_factor), 1, stride=1, padding=0, bias=bias),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, x):
        x = self.up(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(DownSample, self).__init__()

        self.down = nn.Sequential(nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
                                nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, bias=bias))

    def forward(self, x):
        x = self.down(x)
        return x
    
class MSF(nn.Module):
    def __init__(self, in_channels=64, reduction=8, bias=False):
        super(MSF, self).__init__()
        
        self.feat_ext1 = nn.Sequential(*[RGCAB(in_channels, 2, reduction) for _ in range(2)])
        
        self.down1 = DownSample(in_channels, chan_factor=1.5)
        self.feat_ext2 = nn.Sequential(*[RGCAB(int(in_channels*1.5), 2, reduction) for _ in range(2)])
        
        self.down2 = DownSample(int(in_channels*1.5), chan_factor=1.5)
        self.feat_ext3 = nn.Sequential(*[RGCAB(int(in_channels*1.5*1.5), 2, reduction) for _ in range(1)])
               
        self.up2 = UpSample(int(in_channels*1.5*1.5), chan_factor=1.5)
        self.feat_ext5 = nn.Sequential(*[RGCAB(int(in_channels*1.5), 2, reduction) for _ in range(2)])
        
        self.up1 = UpSample(int(in_channels*1.5), chan_factor=1.5)
        self.feat_ext6 = nn.Sequential(*[RGCAB(in_channels, 2, reduction) for _ in range(2)])
        
    def forward(self, x):
        
        x = self.feat_ext1(x) # torch.Size([14, 64, 48, 48])
        
        enc_1 = self.down1(x)
        enc_1 = self.feat_ext2(enc_1) # torch.Size([14, 96, 24, 24])
        
        enc_2 = self.down2(enc_1)
        enc_2 = self.feat_ext3(enc_2) # torch.Size([14, 144, 12, 12])
        
        dec_2 = self.up2(enc_2)
        dec_2 = self.feat_ext5(dec_2 + enc_1) # torch.Size([14, 96, 24, 24])
        
        dec_1 = self.up1(dec_2)
        dec_2 = self.feat_ext6(dec_1 + x) # torch.Size([14, 64, 48, 48])
        
        return dec_2


class MergeBlockUNetDiff(nn.Module):
    def __init__(self, num_features, input_dim, project_dim, 
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

        self.conv2 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=bias))
        self.UNet = nn.Sequential(MSF(num_features))

        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.UNet.parameters():
            param.requires_grad = False
    
    def forward(self, feat):
        burst_feat = self.conv2(feat)       # (14, num_features, H/2, W/2)
        burst_feat = self.UNet(burst_feat)
        
        weight = self.weight_predictor(feat)
        weights_norm = F.softmax(weight, dim=0)
        fused_feat = (feat * weights_norm).sum(dim=0)
        
        return fused_feat.unsqueeze(0)
    
class MergeBlock(nn.Module):
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
        weight = self.weight_predictor(feat)
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
def dcnsrnet_unet_mergediff(alignment_init_dim, reduction, alignment_out_dim, dec_init_conv_dim, 
             dec_num_pre_res_blocks, dec_post_conv_dim, dec_num_post_res_blocks,
             burst_size, upsample_factor=2, activation='relu', icnrinit=False,
             gauss_blur_sd=None, gauss_ksz=3, 
             ):
    
    ebfa = EBFA(num_features=alignment_init_dim, reduction=reduction)
    fusion = MergeBlockUNetDiff(num_features=alignment_init_dim, input_dim=64, project_dim=64)
    decoder = ResPixShuffleConv(alignment_out_dim, dec_init_conv_dim, dec_num_pre_res_blocks,
                                dec_post_conv_dim, dec_num_post_res_blocks,
                                upsample_factor=upsample_factor, activation=activation,
                                gauss_blur_sd=gauss_blur_sd, icnrinit=icnrinit,
                                gauss_ksz=gauss_ksz)
    
    net = DCNSRNet(alignment=ebfa, fusion=fusion, decoder = decoder)
    
    return net