import torch
import torch.nn.functional as F
import numpy as np
import cv2

from implicit_seg.functional import Reconstruction3D

resolutions = [
    (12+1, 20+1, 12+1),
    (24+1, 40+1, 24+1),
    (48+1, 80+1, 48+1),
    (96+1, 160+1, 96+1),
    (192+1, 320+1, 192+1),
]
num_points = [
    None, 
    13*21*13, 
    13*21*13*4, 
    13*21*13*16, 
    13*21*13*64, 
]
align_corners = False

def query_func(tensor, points):
    """
    [Essential!] here align_corners should be same
    with how you process gt through interpolation.
    tensor: (bz, 1, H, W, D)
    points: [bz,] list of (N, 3)
    """
    bz = len(points)
    occupancys = [ 
        F.grid_sample(
            tensor[i].unsqueeze(0), 
            points[i].view(1, 1, 1, -1, 3),
            mode="bilinear",
            align_corners=align_corners,
            padding_mode="border", # to align with F.interpolate
        ).squeeze().unsqueeze(0) for i in range(bz)
    ]
    return occupancys

if __name__ == "__main__":  
    import tqdm  
    # gt
    query_sdfs = torch.load(
        "../data/rp_amber_posed_028_sdf.pth").to("cuda:0").float() # [1, 1, H, W, D]

    # infer
    engine = Reconstruction3D(
        query_func = query_func, 
        b_min = [[-1.0, -1.0, -1.0]],
        b_max = [[1.0, 1.0, 1.0]],
        resolutions = resolutions,
        num_points = num_points,
        align_corners = align_corners,
        balance_value = 0.,
        device="cuda:0", 
        # visualize_path="../data/"
    )

    with torch.no_grad():
        for _ in tqdm.tqdm(range(100)):
            sdfs = engine.forward(tensor=query_sdfs)
    
    