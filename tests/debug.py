import torch
import torch.nn.functional as F

input = torch.tensor([[[
    [ 1.,  2.,  3.,  4.],
    [ 12.,  1.,  1.,  5.],
    [ 11.,  1.,  1.,  6.],
    [ 10.,  9.,  8.,  7.]
]]])


print (
    F.interpolate(input, (5, 5), mode="bilinear", align_corners=False),
    F.grid_sample(input, torch.tensor([[[[-1+1/5 + 2/5, -1+1/5 + 2/5]]]]).float(), mode="bilinear", align_corners=False, padding_mode="border")
)