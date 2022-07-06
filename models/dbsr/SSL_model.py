import models.layers.blocks as blocks
import torch.nn as nn
import torch.nn.functional as F
import torch


class Degrade(nn.Module):
    def __init__(self, down_scale):
        super().__init__()
        self.down_scale = down_scale
    
    def forward(self, im, kernel):
        _, kc, ksize, _ = kernel.size()
        psize = ksize // 2
        assert kc == 1, "only support kc=1!"
        
        if len(im.shape) == 4:
            b, c, h, w = im.size()
            # blur
            im = F.pad(im, (psize, psize, psize, psize), mode='replicate')
            blur_list = []
            for i in range(b):
                blur_list.append(self.conv_func(im[i:i + 1, :, :, :], kernel[i:i + 1, :, :, :]))
            blur = torch.cat(blur_list, dim=0)
            blur = blur[:, :, psize:-psize, psize:-psize]

            # down
            blurdown = blur[:, :, ::self.down_scale, ::self.down_scale]
        elif len(im.shape) == 5:
            b, seq, c, h, w = im.size()
            # blur
            blur = torch.empty((0, b, c, h+2*psize, w+2*psize)).to('cuda:0')
            for i in range(b):
                image = im[i, :, :, :, :]
                image = F.pad(image, (psize, psize, psize, psize), mode='replicate')
                burst_list = []
                for j in range(seq):
                    burst_list.append(self.conv_func(image[j:j + 1, :, :, :], kernel[i:i + 1, :, :, :]))
                burst = torch.cat(burst_list, dim=0)
                blur = torch.vstack((blur, torch.unsqueeze(burst, 0)))
            blur = blur[:, :, :, psize:-psize, psize:-psize]

            # down
            blurdown = blur[:, :, :, ::self.down_scale, ::self.down_scale]

        return blurdown

    def conv_func(self, input, kernel, padding='same'):
        b, c, h, w = input.size()
        assert b == 1, "only support b=1!"
        _, _, ksize, ksize = kernel.size()
        if padding == 'same':
            pad = ksize // 2
        elif padding == 'valid':
            pad = 0
        else:
            raise Exception("not support padding flag!")

        conv_result_list = []
        for i in range(c):
            conv_result_list.append(F.conv2d(input[:, i:i + 1, :, :], kernel, bias=None, stride=1, padding=pad))
        conv_result = torch.cat(conv_result_list, dim=1)
        return conv_result

    # def forward(self, im, kernel):
    #     if len(im.shape) == 4:
    #         return F.interpolate(im,
    #                              scale_factor=(1 / self.down_scale,
    #                                            1 / self.down_scale))
    #     elif len(im.shape) == 5:
    #         return F.interpolate(im,
    #                              scale_factor=(1, 1 / self.down_scale,
    #                                            1 / self.down_scale))


class Projection(nn.Module):
    def __init__(self, dim, proj_dim):
        super().__init__()
        self.dim = dim
        self.projection_size = proj_dim

        self.proj_layers = nn.Sequential(
            nn.Linear(dim, proj_dim*4),
            nn.BatchNorm1d(proj_dim*4),
            nn.ReLU(),
            nn.Linear(proj_dim*4, proj_dim)
        )

    def forward(self, im):
        return self.proj_layers(im.view(im.shape[0], -1))

class Predictor(nn.Module):
    def __init__(self, dim, pred_dim):
        super().__init__()
        self.dim = dim
        self.projection_size = pred_dim

        self.pred_layers = nn.Sequential(
            nn.Linear(pred_dim, pred_dim*4),
            nn.BatchNorm1d(pred_dim*4),
            nn.ReLU(),
            nn.Linear(pred_dim*4, pred_dim)
        )

    def forward(self, im):
        return self.pred_layers(im)