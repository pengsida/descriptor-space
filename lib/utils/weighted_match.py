import numpy as np
from numpy.linalg import pinv
import torch

def weighted_match(descs0, descs1):
    """

    :param descs0: (N1, 2, D)
    :param descs1: (N2, 2, D)
    :return: (N1,), indices
    """
    if isinstance(descs0, torch.Tensor):
        descs0 = descs0.detach().cpu().numpy()
        descs1 = descs1.detach().cpu().numpy()
    N1, _, _ = descs0.shape
    N2, _, _ = descs1.shape
    indices = np.zeros(N1)
    
    for i in range(N1):
        min = 100
        for j in range(N2):
            w0, w1, w2, w3 = weight_descriptors(descs0[i, 0], descs0[i, 1], descs1[j, 0], descs1[j, 1])
            desc0 = w0 * descs0[i, 0] + w1 * descs0[i, 1]
            desc1 = w2 * descs1[j, 0] + w3 * descs1[j, 1]
            dist = np.linalg.norm(desc0 - desc1)
            if dist < min:
                indices[i] = j
                min = dist
    
    return indices


def quadprog_lag(H, c, A=None, b=None):
    """
    Solve quadratic programming with equality constraints.

    Arguments:

        min 1/2 x'*H*x + c'x
        s.t. Ax = b

        'A' can be None, in which case there are no constraints

    Returns:
        Let A be of shape (M, N)

        - x: solution, shape (N,)
        - lamb: multiplier, shape (M)
    """
    
    if A is None:
        # trivial
        x = pinv(H).dot(c)
        lamb = None
    else:
        Hinv = pinv(H)
        temp = b + A.dot(Hinv).dot(c)
        lamb = pinv(A.dot(Hinv).dot(A.T)).dot(temp)
        x = Hinv.dot(A.T.dot(lamb) - c)
    
    return x, lamb


def quadprog(H, c, x0, Ae=None, be=None, Ai=None, bi=None):
    """
    Solve general quadratic programming using active set method

    Arguments:

        min 1/2*x'*H*x + c'x
        s.t. Ae*x = be
             Ai*x >= bi

        - x0: the initial solution

    Returns:

        - x: the solution
        - status: 'success', 'running'
    """
    
    epsilon = 1e-9
    error = 1e-6
    max_iter = 1000
    
    status = 'running'
    x = x0
    
    if Ai is None:
        return quadprog_lag(H, c, Ae, be)[0], 'success'
    
    # current active set
    active = np.zeros_like(bi, dtype=np.bool)
    # condition: Ai.dot(x) == 0
    active_cond = Ai.dot(x) < bi + epsilon
    active[active_cond] = True
    
    for k in range(max_iter):
        # construct subproblem constraints
        As = Ai[active]
        bs = bi[active]
        if not Ae is None:
            As = np.vstack([As, Ae])
            bs = np.hstack([bs, be])
        
        g = H.dot(x) + c
        # find direction
        d, lamb = quadprog_lag(H, g, As, np.zeros_like(bs))
        
        if np.linalg.norm(d) <= error:
            # in this case, test feasibility
            
            lamb = lamb[:np.sum(active)]
            if lamb is None or lamb.shape[0] == 0:
                status = 'success'
                break
            # find the index
            min_idx = lamb.argmin()
            lamb_min = lamb[min_idx]
            
            if lamb_min >= 0:
                status = 'success'
                break
            
            # otherwise, delete it from active set
            min_idx = active.nonzero()[0][min_idx]
            active[min_idx] = False
        else:
            # d is not zero, find the appropriate step size
            alpha = 1.0
            for i in range(len(Ai)):
                if active[i] == False and Ai[i].dot(d) < 0:
                    temp = (bi[i] - Ai[i].dot(x)) / (Ai[i].dot(d))
                    if temp < alpha:
                        alpha = temp
                        ti = i
            
            x = x + alpha * d
            if alpha < 1:
                active[ti] = True
    
    return x, status


def weight_descriptors(descs0, descs1, descs2, descs3, method='std'):
    """
    Arguments:
        - descs0,1,2,3: (D, )
    Returns:
        - w0, w1, w2, w3, weights
    """
    
    H = np.vstack([descs0, descs1, -descs2, -descs3]).T
    H = H.T.dot(H)
    c = np.zeros(H.shape[0])
    
    # constraint
    Ae = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    be = np.array([1, 1])
    Ai = np.eye(4)
    bi = np.zeros(4)
    x0 = np.array([1, 0, 1, 0])
    
    def tofloat(*args):
        return [x.astype(np.float) for x in args]
    
    [H, c, Ae, be, Ai, bi, x0] = tofloat(H, c, Ae, be, Ai, bi, x0)
    
    if method == 'std':
        import qpsolvers
        Ai = -Ai
        bi = -bi
        x = qpsolvers.solve_qp(H, c, Ai, bi, Ae, be)
    else:
        x, status = quadprog(H, c, x0, Ae, be, Ai, bi)
    
    # diff = descs0 * x[0] + descs1 * x[1] - descs2 * x[2] - descs3 * x[3]
    
    return x


def set2set(descs0, descs1, descs2, descs3):
    # (2, D)
    desc0 = np.vstack([descs0, descs1])
    desc1 = np.vstack([descs2, descs3])
    
    diff = desc0[None] - desc1[:, None]
    dist_mtx = np.linalg.norm(diff, axis=2)
    min_dist = dist_mtx.min()
    
    return min_dist



if __name__ == '__main__':
    # a = np.array([1, 2, 3, 4, 5])
    # np.random.seed(233)
    for i in range(10):
        a = np.random.rand(5)
        b = np.random.rand(5)
        c = np.random.rand(5)
        d = np.random.rand(5)
        x, dist0 = weight_descriptors(a, b, c, d, 'std')
        dist1 = set2set(a, b, c, d)
        print(dist0, dist1)
        import time
        
        time.sleep(0.1)
    # weight_descriptors(a, b, c, d, 'my')
    
    # N1 = 1000
    # N2 = 1200
    # D = 128
    # d1 = np.random.randn(N1, 2, 128)
    # d2 = np.random.randn(N2, 2, 128)
    # import time
    # start = time.perf_counter()
    # indices = weighted_match(d1, d2)
    # end = time.perf_counter()
    # print('time: {}s'.format(end - start))

