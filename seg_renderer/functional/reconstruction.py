import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import os

def create_grid3D(min=0, max=256, steps=17, device="cuda:0"):
    arrange = torch.linspace(min, max, steps).long().to(device)
    coords = torch.stack(torch.meshgrid([
        arrange, arrange, arrange
    ])) # [3, 17, 17, 17]
    coords = coords.view(3, -1).t() # [N, 3]
    return coords

def build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1):
    smooth_conv = torch.nn.Conv3d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, padding=padding
    )
    smooth_conv.weight.data = torch.ones(
        (kernel_size, kernel_size, kernel_size), 
        dtype=torch.float32
    ).reshape(in_channels, out_channels, kernel_size, kernel_size, kernel_size) / (kernel_size**3)
    smooth_conv.bias.data = torch.zeros(out_channels)
    return smooth_conv

class Reconstruction3D(nn.Module):
    def __init__(self, query_func, b_min, b_max, resolutions, channels=1, device="cuda:0"):
        super().__init__()
        self.query_func = query_func
        self.b_min = torch.tensor(b_min).float().to(device).unsqueeze(1) #[bz, 1, 3]
        self.b_max = torch.tensor(b_max).float().to(device).unsqueeze(1) #[bz, 1, 3]
        self.resolutions = resolutions
        self.device = device
        self.batchsize = self.b_min.size(0)
        self.channels = channels

        for resolution in resolutions:
            assert resolution % 2 == 1, \
            f"resolution {resolution} need to be odd becuase of align_corner." 

        # init first resolution
        self.init_coords = create_grid3D(
            0, resolutions[-1]-1, steps=resolutions[0], device=self.device) #[N, 3]
        self.init_coords = self.init_coords.unsqueeze(0).repeat(
            self.batchsize, 1, 1) #[bz, N, 3]


    def forward(self, **kwargs):
        # output occupancy field would be:
        # (bz, C, res, res, res)
        
        for resolution in self.resolutions:
            stride = (self.resolutions[-1] - 1) / (resolution - 1)

            # first step
            if resolution == self.resolutions[0]:
                coords = self.init_coords.clone()

                # normalize coords to fit in [b_min, b_max]
                coords3D = coords.float() / (self.resolutions[-1] - 1) 
                coords3D = coords3D * (self.b_max - self.b_min) + self.b_min

                # query function
                occupancys = self.query_func(**kwargs, points=coords3D)
                if type(occupancys) is list:
                    occupancys = torch.stack(occupancys) #[bz, N, 3]
                occupancys = occupancys.view(
                    self.batchsize, resolution, resolution, resolution, 3)
        
            


        print (len(occupancys), occupancys[0].shape)






class Reconstruction2D():
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        pass
