import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .utils import (
    create_grid2D,
    calculate_uncertainty,
    get_uncertain_point_coords_on_grid2D,
    plot_mask2D,
)

class Reconstruction2D(nn.Module):
    def __init__(self, 
                 query_func, b_min, b_max, resolutions, num_points, 
                 channels=1, balance_value=0.5, device="cuda:0", 
                 visualize_path=None):
        super().__init__()
        self.query_func = query_func
        self.b_min = torch.tensor(b_min).float().to(device).unsqueeze(1) #[bz, 1, 2]
        self.b_max = torch.tensor(b_max).float().to(device).unsqueeze(1) #[bz, 1, 2]
        self.resolutions = resolutions
        self.num_points = num_points
        self.device = device
        self.batchsize = self.b_min.size(0)
        self.balance_value = balance_value
        self.channels = channels; assert channels == 1
        self.visualize_path = visualize_path
        if self.visualize_path:
            os.makedirs(visualize_path, exist_ok=True)

        for resolution in resolutions:
            assert resolution % 2 == 1, \
            f"resolution {resolution} need to be odd becuase of align_corner." 

        # init first resolution
        self.init_coords = create_grid2D(
            0, resolutions[-1]-1, steps=resolutions[0], device=self.device) #[N, 2]
        self.init_coords = self.init_coords.unsqueeze(0).repeat(
            self.batchsize, 1, 1) #[bz, N, 2]

    def batch_eval(self, resolution, coords, align_corners=False, **kwargs):
        """
        coords: in the coordinates of last resolution
        align_corners: same with how you process gt. (grid_sample / interpolate) 
        **kwargs: for query_func
        """
        # normalize coords to fit in [b_min, b_max]
        if align_corners:
            coords2D = coords.float() / (self.resolutions[-1] - 1) 
        else:
            step = 1.0 / self.resolutions[-1]
            coords2D = coords.float() / self.resolutions[-1] + step / 2.0
        coords2D = coords2D * (self.b_max - self.b_min) + self.b_min
        # query function
        occupancys = self.query_func(**kwargs, points=coords2D)
        if type(occupancys) is list:
            occupancys = torch.stack(occupancys) #[bz, C, N]
        assert len(occupancys.size()) == 3, \
            "query_func should return a occupancy with shape of [bz, C, N]"
        return occupancys

    def forward(self, **kwargs):
        """
        output occupancy field would be:
        (bz, C, res, res)
        """
        for resolution, num_pt in zip(self.resolutions, self.num_points):
            stride = (self.resolutions[-1] - 1) / (resolution - 1)

            # first step
            if resolution == self.resolutions[0]:
                coords = self.init_coords.clone() # torch.long 
                occupancys = self.batch_eval(resolution, coords, **kwargs, align_corners=False)
                occupancys = occupancys.view(
                    self.batchsize, self.channels, resolution, resolution)

                if self.visualize_path:
                    final = F.interpolate(
                        occupancys.float(), size=self.resolutions[-1], 
                        mode="bilinear", align_corners=True)
                    x = coords[0, :, 0].to("cpu")
                    y = coords[0, :, 1].to("cpu")
                    save_path = os.path.join(
                        self.visualize_path, f"recon2D_res{resolution}.png")
                    plot_mask2D(
                        final[0, 0].to("cpu"), save_path, point_coords=(x, y))

            else:
                occupancys = F.interpolate(
                    occupancys.float(), size=resolution, mode="bilinear", align_corners=True)
                
                if not num_pt > 0:
                    continue

                uncertainty = calculate_uncertainty(occupancys, balance_value=self.balance_value)
                point_indices, point_coords = get_uncertain_point_coords_on_grid2D(
                    uncertainty, num_points=num_pt)
                
                coords = point_coords * stride
                occupancys_topk = self.batch_eval(resolution, coords, **kwargs, align_corners=False)
                
                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C, H * W)
                    .scatter_(2, point_indices, occupancys_topk)
                    .view(R, C, H, W)
                )

                if self.visualize_path:
                    final = F.interpolate(
                        occupancys.float(), size=self.resolutions[-1], 
                        mode="bilinear", align_corners=True)
                    x = coords[0, :, 0].to("cpu")
                    y = coords[0, :, 1].to("cpu")
                    save_path = os.path.join(
                        self.visualize_path, f"recon2D_res{resolution}.png")
                    plot_mask2D(
                        final[0, 0].to("cpu"), save_path, point_coords=(x, y))

        return occupancys
