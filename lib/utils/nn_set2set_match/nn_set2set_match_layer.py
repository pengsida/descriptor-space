import lib.utils.nn_set2set_match.nn_set2set_match as nn_set2set_match
import torch
import numpy as np


def nn_linear_match_v1_cuda(descs0, descs1):
    """
    descs0: [B, N1, 2, D]
    descs1: [B, N2, 2, D]
    """
    descs0 = descs0.float().contiguous()
    descs1 = descs1.float().contiguous()
    
    # allocate space
    b, n1 = descs0.shape[:2]
    idxs = torch.zeros([b, n1], dtype=torch.int32, device=descs0.device).contiguous()
    
    nn_set2set_match.nn_linear_match_v1(descs0, descs1, idxs)
    return idxs
def nn_linear_match_cuda(descs0, descs1):
    """
    descs0: [B, N1, 2, D]
    descs1: [B, N2, 2, D]
    """
    descs0 = descs0.float().contiguous()
    descs1 = descs1.float().contiguous()

    # allocate space
    b, n1 = descs0.shape[:2]
    idxs = torch.zeros([b, n1], dtype=torch.int32, device=descs0.device).contiguous()

    nn_set2set_match.nn_linear_match(descs0, descs1, idxs)
    return idxs


def nn_set2set_match_cuda(descs0, descs1):
    """
    descs0: [B, N1, 2, D]
    descs1: [B, N2, 2, D]
    """
    descs0 = descs0.float().contiguous()
    descs1 = descs1.float().contiguous()

    # allocate space
    b, n1 = descs0.shape[:2]
    idxs = torch.zeros([b, n1], dtype=torch.int32, device=descs0.device).contiguous()

    nn_set2set_match.nn_set2set_match(descs0, descs1, idxs)
    return idxs


def nn_set2set_match_v1_cuda(descs0, descs1):
    """
    descs0: [B, N1, 2, D]
    descs1: [B, N2, 2, D]
    """
    descs0 = descs0.float().contiguous()
    descs1 = descs1.float().contiguous()

    # allocate space
    b, n1 = descs0.shape[:2]
    idxs = torch.zeros([b, n1], dtype=torch.int32, device=descs0.device).contiguous()
    scale_idxs = torch.zeros([b, n1], dtype=torch.int32, device=descs0.device).contiguous()

    nn_set2set_match.nn_set2set_match_v1(descs0, descs1, idxs, scale_idxs)
    return idxs, scale_idxs


def nn_set2set_match_numpy(descs0, descs1):
    """
    descs0: [N1, 2, D]
    descs1: [N2, 2, D]
    """
    n1, _, d = descs0.shape
    n2 = descs1.shape[0]
    diff = descs0.reshape(n1, 1, 2, 1, d) - descs1.reshape(1, n2, 1, 2, d)
    dis_mat = np.linalg.norm(diff, axis=4)
    dis_mat = dis_mat.reshape(n1, n2, 4)
    dis_mat = np.min(dis_mat, axis=2)
    return np.argmin(dis_mat, axis=1)


def nn_match_cuda(descs0, descs1):
    """
    descs0: [B, N1, D]
    descs1: [B, N2, D]
    """
    descs0 = descs0.float().contiguous()
    descs1 = descs1.float().contiguous()

    # allocate space
    b, n1 = descs0.shape[:2]
    idxs = torch.zeros([b, n1], dtype=torch.int32, device=descs0.device).contiguous()

    nn_set2set_match.nn_match(descs0, descs1, idxs)
    return idxs
