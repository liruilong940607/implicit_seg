import torch

def create_grid3D(min, max, steps, device="cuda:0"):
    arrange = torch.linspace(min, max, steps).long().to(device)
    coords = torch.stack(torch.meshgrid([
        arrange, arrange, arrange
    ])) # [3, steps, steps, steps]
    coords = coords.view(3, -1).t() # [N, 3]
    return coords

def create_grid2D(min, max, steps, device="cuda:0"):
    arrange = torch.linspace(min, max, steps).long().to(device)
    coords = torch.stack(torch.meshgrid([
        arrange, arrange
    ])) # [2, steps, steps]
    coords = coords.view(2, -1).t() # [N, 2]
    return coords

def get_uncertain_point_coords_on_grid3D(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W, D) that contains uncertainty
            values for a set of points on a regular H x W x D grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W x D) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 3) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W x D grid.
    """
    R, _, H, W, D = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)
    d_step = 1.0 / float(D)

    num_points = min(H * W * D, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W * D), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 3, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = h_step / 2.0 + (point_indices // (W * D)).to(torch.float) * h_step
    point_coords[:, :, 1] = w_step / 2.0 + (point_indices % (W * D) // D).to(torch.float) * w_step
    point_coords[:, :, 2] = d_step / 2.0 + (point_indices % D).to(torch.float) * d_step
    return point_indices, point_coords

def get_uncertain_point_coords_on_grid2D(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords

def calculate_uncertainty(logits, classes=None, balance_value=0.5):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -torch.abs(gt_class_logits - balance_value)
