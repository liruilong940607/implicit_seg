import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import seg_renderer.cuda.reconstruction as reconstruction_cuda


class ReconstructionFunction(Function):
    @staticmethod
    def forward(ctx, image_feature, eval_func, **kwargs):
        ctx.save_for_backward(None)
        return None, None

    @staticmethod
    def backward(ctx, grad_occupancy):
        return None, None

def reconstruction(image_feature, eval_func, **kwargs):
    if image_feature.device == "cpu":
        raise TypeError('seg_renderer module supports only cuda Tensors')

    return ReconstructionFunction.apply(image_feature, eval_func)