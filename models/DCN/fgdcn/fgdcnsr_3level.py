import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
import models.layers.blocks as blocks
from admin.model_constructor import model_constructor
from models.layers.upsampling import PixShuffleUpsampler
from models.alignment.pwcnet import PWCNet
from admin.environment import env_settings
import os, math

from DCNv2.dcn_v2 import DCN_sep as DCN, FlowGuidedDCN, InsideFlowGuidedDCN


class FlowGuidedPCDAlign(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels. [From EDVR]
    '''

    def __init__(self, nf=64, groups=8):
        super(FlowGuidedPCDAlign, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)# concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)    # concat for fea

        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, nbr_fea_warped_l, ref_fea_l, flows_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_warped_l[2], ref_fea_l[2], flows_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset, flows_l[2]))
        # L2
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = torch.cat([nbr_fea_warped_l[1], ref_fea_l[1], flows_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset*2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset, flows_l[1])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = torch.cat([nbr_fea_warped_l[0], ref_fea_l[0], flows_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset, flows_l[0])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))

        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.cas_dcnpack(L1_fea, offset)

        return L1_fea

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
        base_feat_proj = feat.mean(dim=1, keepdim=True)
        feat_diff = feat - base_feat_proj

        weight = torch.zeros_like(feat_diff)
        
        for i in range(feat_diff.shape[1]):
            weight[:, i, ...] = self.weight_predictor(feat[:, i, ...])
        weights_norm = F.softmax(weight, dim=1)
        fused_feat = (feat * weights_norm).sum(dim=1)
        
        return fused_feat


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


class FGDCNSRNet(nn.Module):
    def __init__(self, alignment, fusion, decoder):
        super().__init__()
        self.center = 0

        num_in_ch = 4
        embed_dim = 96
        num_feat = 64

        spynet_path='/tmp2/tonytseng/deep-rawburst-sr/models/DCN/fgdcn/.pretrained_models/spynet_sintel_final-3d2a1287.pth'
        self.spynet = SpyNet(spynet_path, [3, 4, 5])

        self.align = alignment
        self.fusion = fusion
        self.decoder = decoder

        self.conv_flow = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.flow_ps = nn.PixelShuffle(2)

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch*(1+2*0), embed_dim, 3, 1, 1, bias=True)

        WARB = functools.partial(WideActResBlock, nf=embed_dim)
        self.feature_extraction = make_layer(WARB, 5)

        self.conv_after_pre_layer = nn.Conv2d(embed_dim, num_feat*4, 3, 1, 1, bias=True)
        self.mid_ps = nn.PixelShuffle(2)

        self.fea_L2_conv1 = nn.Conv2d(num_feat, num_feat*2, 3, 2, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(num_feat*2, num_feat*4, 3, 2, 1, bias=True)
        ##################################################################################################
        ################################## 2. Feature enhanced PCD Align #################################
        self.toplayer = nn.Conv2d(num_feat*4, num_feat, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(num_feat*2, num_feat, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(num_feat*1, num_feat, kernel_size=1, stride=1, padding=0)

        ######################################################################################################
        ################################ 5, high quality image reconstruction ################################


        #### skip #############
        self.skip_pixel_shuffle = nn.PixelShuffle(2)
        self.skipup1 = nn.Conv2d(num_in_ch//4, num_feat * 4, 3, 1, 1, bias=True) ####
        self.skipup2 = nn.Conv2d(num_feat, 3 * 4, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### skip module ########
        skip1 = self.lrelu2(self.skip_pixel_shuffle(self.skipup1(self.skip_pixel_shuffle(x_center))))
        skip2 = self.skip_pixel_shuffle(self.skipup2(skip1))

        x_ = self.conv_flow(self.flow_ps(x.view(B*N, C, H, W))).view(B, N, -1, H*2, W*2)
        
        # calculate flows
        ref_flows = self.get_ref_flows(x_)

        #### extract LR features
        x = self.lrelu(self.conv_first(x.view(B*N, -1, H, W)))

        L1_fea = self.mid_ps(self.conv_after_pre_layer(self.feature_extraction(x)))
        _, _, H, W = L1_fea.size()

        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))

        # FPN enhance features
        L3_fea = self.lrelu(self.toplayer(L3_fea))
        L2_fea = self.smooth1(self._upsample_add(L3_fea, self.latlayer1(L2_fea)))
        L1_fea = self.smooth2(self._upsample_add(L2_fea, self.latlayer2(L1_fea)))

        L1_fea = L1_fea.view(B, N, -1, H, W).contiguous()
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2 ).contiguous()
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4).contiguous()

        #### PCD align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), 
            L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), 
                L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            flows_l = [
                ref_flows[0][:, i, :, :, :].clone(), 
                ref_flows[1][:, i, :, :, :].clone(), 
                ref_flows[2][:, i, :, :, :].clone()
            ]
            # print(nbr_fea_l[0].shape, flows_l[0].shape)
            nbr_warped_l = [
                flow_warp(nbr_fea_l[0], flows_l[0].permute(0, 2, 3, 1), 'bilinear'),
                flow_warp(nbr_fea_l[1], flows_l[1].permute(0, 2, 3, 1), 'bilinear'),
                flow_warp(nbr_fea_l[2], flows_l[2].permute(0, 2, 3, 1), 'bilinear')
            ]
            aligned_fea.append(self.align(nbr_fea_l, nbr_warped_l, ref_fea_l, flows_l))

        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W] --> [B, T, C, H, W]

        out_fus = self.fusion(aligned_fea)
        out_dec = self.decoder(out_fus)

        return out_dec, out_fus
    
    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y
    
    def get_ref_flows(self, x):
        '''Get flow between frames ref and other'''

        b, n, c, h, w = x.size()
        x_nbr = x.reshape(-1, c, h, w)
        x_ref = x[:, self.center:self.center+1, :, :, :].repeat(1, n, 1, 1, 1).reshape(-1, c, h, w)

        # backward
        flows = self.spynet(x_ref, x_nbr)
        flows_list = [flow.view(b, n, 2, h // (2 ** (i)), w // (2 ** (i))) for flow, i in
                          zip(flows, range(3))]

        return flows_list


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        # ref = [ref]
        # supp = [supp]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2**(5-level) # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h//scale, w//scale), mode='bilinear', align_corners=False)
                flow_out[:, 0, :, :] *= float(w//scale) / float(w_floor//scale)
                flow_out[:, 1, :, :] *= float(h//scale) / float(h_floor//scale)
                if torch.abs(flow_out).mean() > 200:
                    print(f"level {level}, flow > 200: {torch.abs(flow_out).mean():.4f}")
                    # return None
                    flow_out.clamp(-250, 250)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


@model_constructor
def fgdcnnet(num_features, reduction, alignment_out_dim, dec_init_conv_dim, 
             dec_num_pre_res_blocks, dec_post_conv_dim, dec_num_post_res_blocks,
             burst_size, upsample_factor=2, activation='relu', icnrinit=False,
             gauss_blur_sd=None, gauss_ksz=3, 
             ):
    # alignment_net = PWCNet(load_pretrained=True,
    #                     weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))
    flowPCD = FlowGuidedPCDAlign(nf=num_features, groups=reduction)
    fusion = MergeBlockDiff(input_dim=64, project_dim=64)
    decoder = ResPixShuffleConv(alignment_out_dim, dec_init_conv_dim, dec_num_pre_res_blocks,
                                dec_post_conv_dim, dec_num_post_res_blocks,
                                upsample_factor=upsample_factor, activation=activation,
                                gauss_blur_sd=gauss_blur_sd, icnrinit=icnrinit,
                                gauss_ksz=gauss_ksz)

    net = FGDCNSRNet(alignment=flowPCD, fusion=fusion, decoder=decoder)

    return net

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    x = x.float()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4': # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output

class WideActResBlock(nn.Module):
    def __init__(self, nf=64):
        super(WideActResBlock, self).__init__()
        self.res_scale = 1
        body = []
        expand = 6
        linear = 0.8
        kernel_size = 3
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act=nn.ReLU(True)

        body.append(
            wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(nf*linear), nf, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)