import torch
import torch.nn.functional as F
import numpy as np
import cv2

from implicit_seg.functional import Reconstruction3D

resolutions = [
    (16+1, 16+1, 16+1),
    (32+1, 32+1, 32+1),
    # (64+1, 64+1, 64+1),
    # (128+1, 128+1, 128+1),
    # (256+1, 256+1, 256+1),
]
num_points = [
    None, 
    8000, 
    # 8000, 
    # 8000, 
    # 8000, 
]
clip_mins = [
    None,
    -1e9,
    # -1e9,
    # -1e9,
    # -1e9,
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

    if type(resolutions[-1]) is int:
        final_W, final_H, final_D = resolutions[-1], resolutions[-1], resolutions[-1]
    else:
        final_W, final_H, final_D = resolutions[-1]
    gt = F.interpolate(
        query_sdfs, (final_D, final_H, final_W), mode="trilinear", align_corners=align_corners)
    print ("gt:", gt.shape)

    # infer
    engine = Reconstruction3D(
        query_func = query_func, 
        b_min = [[-1.0, -1.0, -1.0]],
        b_max = [[1.0, 1.0, 1.0]],
        resolutions = resolutions,
        num_points = num_points,
        clip_mins = clip_mins,
        align_corners = align_corners,
        balance_value = 0.,
        device="cuda:0", 
        # visualize_path="../data/"
    )

    with torch.no_grad():
        for _ in tqdm.tqdm(range(1)):
            sdfs = engine.forward(tensor=query_sdfs)
    print (sdfs.shape)
    cv2.imwrite(
       "../data/gen_sdf_sumz.png",
       np.uint8(((sdfs[0, 0]>0).sum(dim=0)>0).float().cpu().numpy() * 255)
    )
    cv2.imwrite(
       "../data/gen_sdf_sumx.png",
       np.uint8(((sdfs[0, 0]>0).sum(dim=2)>0).float().cpu().numpy().transpose() * 255)
    )

    # metric
    intersection = (sdfs > 0.) & (gt > 0.)
    union = (sdfs > 0.) | (gt > 0.)
    iou = intersection.sum().float() / union.sum().float()
    print (f"iou is {iou}")
    