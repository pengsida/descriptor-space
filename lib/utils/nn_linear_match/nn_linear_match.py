import numpy as np
import qpsolvers


def compute_linear_interpol_weight(desc0, desc1):
    """
    desc0: [2, D]
    desc1: [2, D]
    """
    P = np.dot(desc0, desc0.T).astype(np.double)
    q = -np.dot(desc1[0], desc0.T).astype(np.double)
    G = -np.eye(2).astype(np.double)
    h = np.zeros(2).astype(np.double)
    A = np.array([[1., 1.]]).astype(np.double)
    b = np.array([1]).astype(np.double)
    return qpsolvers.solve_qp(P, q, G, h, A, b)


def compute_linear_interpol_distance(desc0, desc1):
    """
    desc0: [2, D]
    desc1: [
    """
    x0 = compute_linear_interpol_weight(desc0, desc1)
    distance0 = np.linalg.norm(np.dot(x0, desc0) - desc1)
    x1 = compute_linear_interpol_weight(desc1, desc0)
    distance1 = np.linalg.norm(np.dot(x1, desc1) - desc0)
    print(x0, x1)
    return min(distance0, distance1)


def nn_linear_interpol_match_numpy(prd_descs0, prd_descs1):
    """
    prd_desc0: [N, 2, D]
    prd_desc1: [N, 2, D]
    """
    n = prd_descs0.shape[0]
    for i in range(n):
        desc0 = prd_descs0[i]
        desc1 = prd_descs1[i]
        print(compute_linear_interpol_distance(desc0, desc1))
        import ipdb; ipdb.set_trace()
