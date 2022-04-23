import models.layers.blocks as blocks
import torch.nn as nn
import torch.nn.functional as F


class Degrade(nn.Module):
    def __init__(self, down_scale):
        super().__init__()
        self.down_scale = down_scale

    def forward(self, im):
        if len(im.shape) == 4:
            return F.interpolate(im,
                                 scale_factor=(1 / self.down_scale,
                                               1 / self.down_scale))
        elif len(im.shape) == 5:
            return F.interpolate(im,
                                 scale_factor=(1, 1 / self.down_scale,
                                               1 / self.down_scale))


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