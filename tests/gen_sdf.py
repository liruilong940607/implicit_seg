
import cv2
import trimesh
import numpy as np
import torch
from scipy.spatial import cKDTree

from implicit_seg.functional import Reconstruction3D

class HoppeSDF:
    def __init__(self, points, normals):
        '''
        The HoppeSDF calculates signed distance towards a predefined oriented point cloud
        http://hhoppe.com/recon.pdf
        For clean and high-resolution pcl data, this is the fastest and accurate approximation of sdf
        :param points: pts
        :param normals: normals
        '''
        self.points = points
        self.normals = normals
        self.kd_tree = cKDTree(self.points)
        self.len = len(self.points)

    def query(self, points):
        # points = points.T
        dists, idx = self.kd_tree.query(points)
        dirs = points - self.points[idx]
        signs = (dirs * self.normals[idx]).sum(axis=1)
        signs = (signs > 0) * 2 - 1
        return - signs * dists

def load_model(path, scale_factor=1.0):
    model = trimesh.load(path)
    verts = model.vertices 
    faces = model.faces
    normals = model.vertex_normals

    b_min = verts.min(axis = 0) 
    b_max = verts.max(axis = 0)
    size = (b_max - b_min) / 2
    center = (b_max + b_min) / 2
    b_min = center - size / scale_factor
    b_max = center + size / scale_factor
    return HoppeSDF(verts, normals), b_min, b_max

def query_sdf_hoppe(models, points):
    """
    models [bz,] list, references for query, such as meshes, feature maps as so on
    points [bz,] list of (n,3) float, list of points in space
    return [bz,] list of (n,) float, occupancy field based on sdf. 1-inside, 0-outside
    """    
    device = None
    if type(points[0]) == torch.Tensor:
        device = points[0].device
        points = [ps.cpu().numpy() for ps in points]

    sdfs = [model.query(ps) for (model, ps) in zip(models, points)]

    if device is not None:
        sdfs = [torch.from_numpy(sdf).to(device).unsqueeze(0) for sdf in sdfs]
    return sdfs

if __name__ == "__main__":
    resolutions = [(80+1, 200+1, 120+1)]
    num_points = [None]
    align_corners = True

    model, b_min, b_max = load_model(
        path="../models/rp_amber_posed_028.obj", scale_factor=0.7)

    # infer
    engine = Reconstruction3D(
        query_func = query_sdf_hoppe, 
        b_min = [b_min,],
        b_max = [b_max,],
        resolutions = resolutions,
        num_points = num_points,
        align_corners = align_corners,
        balance_value = 0.,
        device="cuda:0", 
        # visualize_path="../data/"
    )
    sdfs = engine.forward(models=[model,]) # [1, 1, D, H, W]
    print (sdfs.shape)
    torch.save(sdfs, "../data/rp_amber_posed_028_sdf.pth")

    cv2.imwrite(
       "../data/gen_sdf_sumz.png",
       np.uint8(((sdfs[0, 0]>0).sum(dim=0)>0).float().cpu().numpy() * 255)
    )
    cv2.imwrite(
       "../data/gen_sdf_sumx.png",
       np.uint8(((sdfs[0, 0]>0).sum(dim=2)>0).float().cpu().numpy().transpose() * 255)
    )