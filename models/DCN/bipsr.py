import torch
import torch.nn as nn
from turtle import forward
from torchvision.ops import DeformConv2d
# import models.layers.blocks as blocks
# from models.layers.upsampling import PixShuffleUpsampler
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

class PBFF(nn.Module):
    def __init__(self, num_features, burst_size, bias=False):
        super().__init__()
        ####### Pseudo Burst Feature Fusion
        self.conv2 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=bias))
        # self.conv2 = nn.Sequential(nn.Conv2d(14, num_features, kernel_size=3, padding=1, bias=bias)) # my_bip
        
        ## Multi-scale Feature Extraction
        self.UNet = nn.Sequential(MSF(num_features))
    
    def forward(self, burst_feat):
        ##################################################
        ####### Pseudo Burst Feature Fusion ####################
        ##################################################
        # burst_feat = burst_feat.permute(1,0,2,3).contiguous()
        # burst_feat = self.conv2(burst_feat)       # (num_features, num_features, H/2, W/2)      

        ## Multi-scale Feature Extraction
        burst_feat = self.UNet(burst_feat)        # (num_features, num_features, H/2, W/2)

        return burst_feat


##############################################################################################
######################### Adaptive Group Up-sampling Module ##########################################
##############################################################################################

class UPSL(nn.Module):
    def __init__(self, in_channels, height, reduction=8, bias=False):
        super(UPSL, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(negative_slope=0.2,inplace=True))

        self.convs = nn.ModuleList([])
        for i in range(self.height):
            self.convs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)
        self.up = nn.ConvTranspose2d(in_channels*4, in_channels, 3, stride=2, padding=1, output_padding=1, bias=bias)

    def forward(self, inp_feats):
        batch_size, b, n_feats, H, W = inp_feats.size()
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_Z = self.conv_du(feats_U)
        
        dense_attention = [conv(feats_Z) for conv in self.convs]
        dense_attention = torch.cat(dense_attention, dim=1)
        
        dense_attention = dense_attention.view(batch_size, self.height, n_feats, H, W)
        
        dense_attention = self.softmax(dense_attention)
        
        feats_V = inp_feats*dense_attention
        feats_V = feats_V.view(batch_size, -1, H, W)
        feats_V = self.up(feats_V)
        
        return feats_V


class AGU(nn.Module):
    def __init__(self, num_features, height, reduction=8, bias=False):
        super(AGU, self).__init__()

        ####### Adaptive Group Up-sampling
        self.SKFF1 = UPSL(num_features, height)
        self.SKFF2 = UPSL(num_features, height)
        self.SKFF3 = UPSL(num_features, height)

        ## Output Convolution
        self.conv3 = nn.Sequential(nn.Conv2d(num_features, 3, kernel_size=3, padding=1, bias=bias))
    
    def forward(self, burst_feat):
        ##################################################
        ####### Adaptive Group Up-sampling #####################
        ##################################################

        b, f, H, W = burst_feat.size()
        burst_feat = burst_feat.view(b//4, 4, f, H, W)          # (num_features//4, 4, num_features, H/2, W/2)        
        burst_feat = self.SKFF1(burst_feat)                     # (num_features//4, num_features, H, W)   
        
        b, f, H, W = burst_feat.size()
        burst_feat = burst_feat.view(b//4, 4, f, H, W)          # (num_features//16, 4, num_features, H, W)  
        burst_feat = self.SKFF2(burst_feat)                     # (num_features//16, num_features, 2H, 2W) 
        
        b, f, H, W = burst_feat.size()
        burst_feat = burst_feat.view(b//4, 4, f, H, W)          # (1, 4, num_features, H, W)  
        burst_feat = self.SKFF3(burst_feat)                     # (1, num_features, 4H, 4W) 
        
        ## Output Convolution
        burst_feat = self.conv3(burst_feat)                     # (1, 3, 4H, 4W) 
        
        return burst_feat


class BIPNet(nn.Module):
    """ BIP model"""
    def __init__(self, EBFA, PBFF, AGU):
        super().__init__()

        self.EBFA = EBFA    # Encodes input images and performs alignment
        self.PBFF = PBFF
        self.AGU = AGU      # Decodes the merged embeddings to generate HR RGB image

    def forward(self, im):
        out_enc = self.EBFA(im)
        out_fus = self.PBFF(out_enc)
        out_dec = self.AGU(out_fus)

        return out_dec


@model_constructor
def bipnet(num_features, reduction, burst_size):
    ebfa = EBFA(num_features=num_features, reduction=reduction)
    pbff = PBFF(num_features=num_features, burst_size=burst_size)
    agu = AGU(num_features=num_features, height=4)

    net = BIPNet(EBFA=ebfa, PBFF=pbff, AGU=agu)

    return net