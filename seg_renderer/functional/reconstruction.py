import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

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
    
def calculate_uncertainty(logits, classes=None, balance_value=0.5):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -torch.abs(gt_class_logits - balance_value)

def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W, D) that contains uncertainty
            values for a set of points on a regular H x W x D grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W x D) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 3) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W x D grid.
    """
    R, _, H, W, D = uncertainty_map.shape
    # h_step = 1.0 / float(H)
    # w_step = 1.0 / float(W)
    # d_step = 1.0 / float(D)

    num_points = min(H * W * D, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W * D), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 3, dtype=torch.float, device=uncertainty_map.device)
    # point_coords[:, :, 0] = w_step / 2.0 + (point_indices % (W * D) // D).to(torch.float) * w_step
    # point_coords[:, :, 1] = h_step / 2.0 + (point_indices // (W * D)).to(torch.float) * h_step
    # point_coords[:, :, 2] = d_step / 2.0 + (point_indices % D).to(torch.float) * d_step
    point_coords[:, :, 0] = (point_indices // (W * D)).to(torch.float)
    point_coords[:, :, 1] = (point_indices % (W * D) // D).to(torch.float)
    point_coords[:, :, 2] = (point_indices % D).to(torch.float)
    return point_indices, point_coords

class Reconstruction3D(nn.Module):
    def __init__(self, query_func, b_min, b_max, resolutions, num_points, channels=1, device="cuda:0"):
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


    def forward(self, **kwargs):
        # output occupancy field would be:
        # (bz, C, res, res, res)
        
        for resolution, num_pt in zip(self.resolutions, self.num_points):
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
                    occupancys = torch.stack(occupancys) #[bz, C, N]
                assert len(occupancys.size()) == 3, \
                    "query_func should return a occupancy with shape of [bz, C, N]"
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
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty, num_points=num_pt)
                
                coords = point_coords * stride
                # normalize coords to fit in [b_min, b_max]
                coords3D = coords.float() / (self.resolutions[-1] - 1) 
                coords3D = coords3D * (self.b_max - self.b_min) + self.b_min

                # query function
                occupancys_topk = self.query_func(**kwargs, points=coords3D)
                if type(occupancys_topk) is list:
                    occupancys_topk = torch.stack(occupancys_topk) #[bz, C, N]
                
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






class Reconstruction2D():
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **kwargs):
        pass
