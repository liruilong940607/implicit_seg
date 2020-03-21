import sys; sys.path.insert(0, "../")
import trimesh
import numpy as np
import torch
from scipy.spatial import cKDTree

from seg_renderer.functional.reconstruction import Reconstruction3D

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
        return signs * dists

def load_model(path="../models/bunny.ply", scale_factor=1.0, hoppe=True):
    model = trimesh.load(path)
    verts = model.vertices * scale_factor
    faces = model.faces
    normals = model.vertex_normals

    b_min = verts.min(axis = 0)
    b_max = verts.max(axis = 0)
    size = b_max - b_min
    b_min = b_min - 0.1 * size
    b_max = b_max + 0.1 * size
    print (f"===> load {path} with scale factor {scale_factor}")
    print (f"b_min: {b_min};")
    print (f"b_max: {b_max};")
    if hoppe:
        return HoppeSDF(verts, normals), [b_min, b_max]
    else:
        return trimesh.Trimesh(vertices=verts, faces=faces), [b_min, b_max]

def query_sdf(refs, points):
    """
    refs [bz,] list, references for query, such as meshes, feature maps as so on
    points [bz,] list of (n,3) float, list of points in space
    return [bz,] list of (n,) float, occupancy field based on sdf. 1-inside, 0-outside
    """    
    return [
        trimesh.proximity.signed_distance(ref, ps) \
        for (ref, ps) in zip(refs, points)
    ]

def query_sdf_hoppe(refs, points):
    """
    refs [bz,] list, references for query, such as meshes, feature maps as so on
    points [bz,] list of (n,3) float, list of points in space
    return [bz,] list of (n,) float, occupancy field based on sdf. 1-inside, 0-outside
    """    
    return [
        ref.query(ps) \
        for (ref, ps) in zip(refs, points)
    ]

def query_oc(refs, points):
    """
    refs [bz,] list, references for query, such as meshes, feature maps as so on
    points [bz,] list of (n,3) float, list of points in space
    return [bz,] list of (n,) float, occupancy field based on sdf. 1-inside, 0-outside
    """
    print ("============== query info ===============")
    print (f"type of refs sent to query: {type(refs[0])}; batchsize={len(refs)}")
    print (f"type of points sent to query: {type(points[0])}; shape={points[0].shape}")
    print ("=========================================")

    device = None
    if type(points[0]) == torch.Tensor:
        device = points[0].device
        points = [ps.cpu().numpy() for ps in points]

    if type(refs[0]) == trimesh.Trimesh:
        sdfs = query_sdf(refs, points)
        occupancys = [np.float32(sdf > 0) for sdf in sdfs]
    elif type(refs[0]) == HoppeSDF:
        sdfs = query_sdf_hoppe(refs, points)
        occupancys = [np.float32(sdf > 0) for sdf in sdfs]
    else:
        raise NotImplementedError

    if device is not None:
        occupancys = [torch.from_numpy(oc).to(device) for oc in occupancys]

    return occupancys

models = [
    load_model(scale_factor=1.0), 
    load_model(scale_factor=0.5),
]
refs = [
    model[0] for model in models
]
b_min = [
    model[1][0] for model in models
]
b_max = [
    model[1][1] for model in models
]
points = [
    np.array([[0.0, 0.1, 0.0], [0.0, 0.1, 0.1]]), 
    np.array([[0.0, 0.1, 0.0]])
]
print (
    "===> results \n",
    query_oc(refs, points),
)

engine = Reconstruction3D(
    query_func = query_oc, 
    b_min = b_min,
    b_max = b_max,
    resolutions = [28+1, 56+1, 112+1, 224+1],
    device="cuda:0", 
)

occupancys = engine.forward(refs=refs)