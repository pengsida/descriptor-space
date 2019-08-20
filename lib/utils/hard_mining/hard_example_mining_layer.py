import lib.utils.hard_mining.hard_example_mining as hem
import torch

def hard_example_mining_layer(feats,feats_ref,gt_loc,interval=32,thresh=8):
    '''

    :param feats:       b,f,h,w
    :param feats_ref:   b,n,f
    :param gt_loc:      b,n,2
    :param interval:
    :param thresh:
    :return: b,n,3
    '''
    b,n,f=feats_ref.shape
    feats=feats.permute(0,2,3,1).float().contiguous()
    feats_ref=feats_ref.float().contiguous()
    gt_loc=gt_loc.float().contiguous()
    hard_idxs=torch.zeros([b,n,3],dtype=torch.int32,device=feats.device).contiguous()
    hem.hard_example_mining(feats,feats_ref,gt_loc,hard_idxs,interval,thresh*thresh)
    return hard_idxs

def knn_hard_example_mining_layer(feats,feats_ref,gt_loc,k,interval=32,thresh=8):
    '''

    :param feats:       b,f,h,w
    :param feats_ref:   b,n,f
    :param gt_loc:      b,n,2
    :param interval:
    :param thresh:
    :return: b,n,3
    '''
    b,n,f=feats_ref.shape
    feats=feats.permute(0,2,3,1).float().contiguous()
    feats_ref=feats_ref.float().contiguous()
    gt_loc=gt_loc.float().contiguous()
    hard_idxs=torch.zeros([b,n,k,3],dtype=torch.int32,device=feats.device).contiguous()
    hem.knn_hard_example_mining(feats,feats_ref,gt_loc,hard_idxs,interval,thresh*thresh)
    return hard_idxs

def semi_hard_example_mining_layer(feats,feats_dist,feats_ref,gt_loc,interval=32,thresh=8,margin=0.0):
    '''

    :param feats:       b,f,h,w
    :param feats_dist:  b,n
    :param feats_ref:   b,n,f
    :param gt_loc:      b,n,2
    :param interval:
    :param thresh:
    :param margin:
    :return: b,n,3
    '''
    b,n,f=feats_ref.shape
    feats=feats.permute(0,2,3,1).float().contiguous()
    feats_dist=feats_dist.float().contiguous()
    feats_ref=feats_ref.float().contiguous()
    gt_loc=gt_loc.float().contiguous()
    hard_idxs=torch.zeros([b,n,3],dtype=torch.int32,device=feats.device).contiguous()
    hem.semi_hard_example_mining(feats,feats_dist,feats_ref,gt_loc,hard_idxs,interval,thresh*thresh,margin)
    return hard_idxs

def knn_semi_hard_example_mining_layer(feats,feats_dist,feats_ref,gt_loc,k,interval=32,thresh=8,margin=1.0):
    '''

    :param feats:       b,f,h,w
    :param feats_dist:  b,n
    :param feats_ref:   b,n,f
    :param gt_loc:      b,n,2
    :param interval:
    :param thresh:
    :param margin:
    :return: b,n,3
    '''
    b,n,f=feats_ref.shape
    feats=feats.permute(0,2,3,1).float().contiguous()
    feats_dist=feats_dist.float().contiguous()
    feats_ref=feats_ref.float().contiguous()
    gt_loc=gt_loc.float().contiguous()
    hard_idxs=torch.zeros([b,n,k,3],dtype=torch.int32,device=feats.device).contiguous()
    hem.knn_semi_hard_example_mining(feats,feats_dist,feats_ref,gt_loc,hard_idxs,interval,thresh*thresh,margin)
    return hard_idxs
