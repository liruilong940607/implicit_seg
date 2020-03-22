# import sys; sys.path.insert(0, "../")
import trimesh
import numpy as np

from implicit_seg.functional.reconstruction import Reconstruction2D

## support batch processing.
## support normal calculation.
## support forward and backward.

def load_model(path="../models/bunny.ply", scale_factor=1.0):
    model = trimesh.load(path)
    verts = model.vertices * scale_factor
    faces = model.faces

    b_min = verts.min(axis = 0)
    b_max = verts.max(axis = 0)
    size = b_max - b_min
    b_min = b_min - 0.1 * size
    b_max = b_max + 0.1 * size
    print (f"===> load {path} with scale factor {scale_factor}")
    print (f"b_min: {b_min};")
    print (f"b_max: {b_max};")
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

    if type(refs[0]) == trimesh.Trimesh:
        sdfs = query_sdf(refs, points)
        occupancy = [np.float32(sdf > 0) for sdf in sdfs]
        return occupancy
    else:
        raise NotImplementedError

models = [
    load_model(scale_factor=1.0), 
    load_model(scale_factor=0.5),
]
refs = [
    model[0] for model in models
]
bboxs = [
    model[1] for model in models
]
points = [
    np.array([[0.0, 0.1, 0.0], [0.0, 0.1, 0.1]]), 
    np.array([[0.0, 0.1, 0.0]])
]
print (
    "===> results \n",
    query_sdf(refs, points),
    query_oc(refs, points),
)

def query_func2d(refs, points2d):
    points = [
        np.concatenate([ps, np.zeros_like(ps[:, 0:1])], axis=1) \
        for ps in points2d
    ]
    return query_oc(refs, points)
bboxs2d = [[bbox[0][0:2], bbox[1][0:2]] for bbox in bboxs]

points = [
    np.array([[0.0, 0.1], [0.0, 0.2]]), 
    np.array([[0.0, 0.1]])
]
print (
    "===> results \n",
    query_func2d(points),
)

engine = Reconstruction2D(
    query_func = query_func2d, 
    bboxs = bboxs2d,
    resolutions = [28, 56, 112, 224],
    device="cuda:0", 
)