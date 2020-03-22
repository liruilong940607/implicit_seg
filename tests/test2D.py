import torch
import torch.nn.functional as F
import numpy as np
import cv2

from implicit_seg.functional import Reconstruction2D

def query_func(tensor, points):
    """
    tensor: (bz, 1, H, W)
    points: [bz,] list of (N, 2)
    """
    bz = len(points)
    occupancys = [ 
        F.grid_sample(
            tensor[i].unsqueeze(0), 
            points[i].view(1, 1, -1, 2),
            align_corners = False,
        ).squeeze().unsqueeze(0) for i in range(bz)
    ]
    return occupancys

if __name__ == "__main__":
    query_mask = torch.from_numpy(
        cv2.imread("../models/image.png", cv2.IMREAD_UNCHANGED)[:, :, 3]
    ).unsqueeze(0).unsqueeze(0).to("cuda:0").float() / 255.0

    engine = Reconstruction2D(
        query_func = query_func, 
        b_min = [[-1.0, -1.0]],
        b_max = [[1.0, 1.0]],
        resolutions = [224+1],
        num_points = [None],
        device="cuda:0", 
    )

    occupancys = engine.forward(tensor=query_mask)
    
    cv2.imwrite(
       "../data/test2D.png",
       np.uint8(occupancys[0, 0].cpu().numpy() * 255)
    )