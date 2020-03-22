import torch
import torch.nn.functional as F
import numpy as np
import cv2

from implicit_seg.functional import Reconstruction2D
from implicit_seg.functional.utils import plot_mask2D

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
    import tqdm
    resolutions = [28+1, 56+1, 112+1, 224+1, 448+1, 896+1]
    num_points = [None, 29**2, 29**2, 29**2, 29**2, 29**2]
    
    # gt
    query_mask = torch.from_numpy(
        cv2.blur(cv2.imread("../models/image.png", cv2.IMREAD_UNCHANGED), (20, 20))[:, :, 3]
    ).unsqueeze(0).unsqueeze(0).to("cuda:0").float() / 255.0

    # infer
    engine = Reconstruction2D(
        query_func = query_func, 
        b_min = [[-1.0, -1.0]],
        b_max = [[1.0, 1.0]],
        resolutions = resolutions,
        num_points = num_points,
        device="cuda:0", 
        # visualize_path="../data/"
    )
    with torch.no_grad():
        for _ in tqdm.tqdm(range(1000)):
            occupancys = engine.forward(tensor=query_mask)

    