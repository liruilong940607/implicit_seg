import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

import numpy as np
import math
import os

from .utils import create_grid3D
from .utils import calculate_uncertainty
from .utils import get_uncertain_point_coords_on_grid3D

class Reconstruction3D(nn.Module):
    def __init__(self, 
                 query_func, b_min, b_max, resolutions, num_points, 
                 channels=1, device="cuda:0"):
        super().__init__()
        self.query_func = query_func
        self.b_min = torch.tensor(b_min).float().to(device).unsqueeze(1) #[bz, 1, 3]
        self.b_max = torch.tensor(b_max).float().to(device).unsqueeze(1) #[bz, 1, 3]
        self.resolutions = resolutions
        self.num_points = num_points
        self.device = device
        self.batchsize = self.b_min.size(0)
        self.channels = channels
        assert channels == 1

        for resolution in resolutions:
            assert resolution % 2 == 1, \
            f"resolution {resolution} need to be odd becuase of align_corner." 

        # init first resolution
        self.init_coords = create_grid3D(
            0, resolutions[-1]-1, steps=resolutions[0], device=self.device) #[N, 3]
        self.init_coords = self.init_coords.unsqueeze(0).repeat(
            self.batchsize, 1, 1) #[bz, N, 3]

    def batch_eval(self, resolution, coords, **kwargs):
        """
        coords: in the coordinates of last resolution
        **kwargs: for query_func
        """
        # normalize coords to fit in [b_min, b_max]
        coords3D = coords.float() / (self.resolutions[-1] - 1) 
        coords3D = coords3D * (self.b_max - self.b_min) + self.b_min

        # query function
        occupancys = self.query_func(**kwargs, points=coords3D)
        if type(occupancys) is list:
            occupancys = torch.stack(occupancys) #[bz, C, N]
        assert len(occupancys.size()) == 3, \
            "query_func should return a occupancy with shape of [bz, C, N]"
        return occupancys

    def forward(self, **kwargs):
        # output occupancy field would be:
        # (bz, C, res, res, res)
        
        for resolution, num_pt in zip(self.resolutions, self.num_points):
            stride = (self.resolutions[-1] - 1) / (resolution - 1)

            # first step
            if resolution == self.resolutions[0]:
                coords = self.init_coords.clone()
                occupancys = self.batch_eval(resolution, coords, **kwargs)
                occupancys = occupancys.view(
                    self.batchsize, self.channels, resolution, resolution, resolution)

                # visualize
                if False:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')

                    voxels = occupancys[0, 0].cpu().numpy() > 0.5
                    voxels = voxels.transpose(0, 2, 1)[::-1, :, :]
                    colors = np.empty(voxels.shape, dtype=object)
                    colors[voxels] = 'red'

                    ax.voxels(voxels, facecolors=colors, edgecolor='k')
                    plt.savefig(f"../data/oc_res{resolution}.png")
            
            else:
                occupancys = F.interpolate(
                    occupancys.float(), size=resolution, mode="trilinear", align_corners=True)
                
                if not num_pt > 0:
                    continue

                uncertainty = calculate_uncertainty(occupancys, balance_value=0.5)
                point_indices, point_coords = get_uncertain_point_coords_on_grid3D(
                    uncertainty, num_points=num_pt)
                
                coords = point_coords * stride
                occupancys_topk = self.batch_eval(resolution, coords, **kwargs)
                
                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W, D = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C, H * W * D)
                    .scatter_(2, point_indices, occupancys_topk)
                    .view(R, C, H, W, D)
                )

                # visualize
                if False:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')

                    voxels = occupancys[0, 0].cpu().numpy() > 0.5
                    voxels = voxels.transpose(0, 2, 1)[::-1, :, :]
                    colors = np.empty(voxels.shape, dtype=object)
                    colors[voxels] = 'red'

                    ax.voxels(voxels, facecolors=colors, edgecolor='k')
                    plt.savefig(f"../data/oc_res{resolution}.png")
                    
                    # coords
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')

                    voxels = torch.zeros_like(occupancys)
                    voxels = (
                        voxels.reshape(R, C, H * W * D)
                        .scatter_(2, point_indices, 1.0)
                        .view(R, C, H, W, D)
                    )
                    voxels = voxels[0, 0].cpu().numpy() > 0
                    voxels = voxels.transpose(0, 2, 1)[::-1, :, :]
                    colors = np.empty(voxels.shape, dtype=object)
                    colors[voxels] = 'red'

                    ax.voxels(voxels, facecolors=colors, edgecolor='k')
                    plt.savefig(f"../data/pt_res{resolution}.png")

        return occupancys
