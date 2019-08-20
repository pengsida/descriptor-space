import numpy as np
import cv2
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from pyflann import FLANN
from tools.pck_eval.descriptor_wrappers import *
from tools.hpatches_eval.hpatches_dataset import *

def evaluate_pck_single(database, model, threshold=10, num_samples=None):
    """
    Evaluate PCK (i.e., percentage of correct keypoints) on a database
    
    :param database: next(iter(database)) should return (img1, img2, H)
    :param model: should have a method "compute(img) -> feats" that computes
                  dense feature
    :param num_sample: number of query points. If None, all matchable points
                       will be used
    :param threshold: threshold in pixel
    :return: average pck
    """

    pbar = tqdm(total=len(database))
    pcks = []
    for i, data in enumerate(database):
        img0, img1, H = data
        (h0, w0), (h1, w1) = img0.shape[:2], img1.shape[:2]
        
        # get flow and matchability
        flow, matchability = find_correspondence((h0, w0), (h1, w1), H)
        
        # compute dense descriptors
        feats0 = model.compute(img0)
        feats1 = model.compute(img1)
        
        # sample query points
        query_points = generate_img_grid(h0, w0)[matchability]
        if num_samples:
            query_points = query_points[np.random.choice(
                query_points.shape[0], num_samples, replace=False)]
        
        
        pck, _ = compute_pck(feats0, feats1, query_points, flow, threshold)
        pcks.append(pck)
        print('{}: {}'.format(i, sum(pcks) / len(pcks)))
        pbar.update()
    

    return sum(pcks) / len(pcks)

def compute_pck(feats_query, feats_target, query_points, flow, threshold):
    
    """
    Compute PCK given a pair of images.
    
    VERY IMPORTANT: each element of query_points should be (x, y), rather than
                    (y, x)
    
    :param feats_query: the feature map of the query image, shape (H, W, D)
    :param feats_target: the feature map of the target image, shape (H0, W0, D)
    :param query_points: query locations, shape (N, 2)
    :param flow: flow from query to target, shape (H, W, 2), where each element
                 is (x, y) location of the target image
    :param H: the mapping from source to target. P in query image corrseponds to
              HP in the target image
    :param correctness threshold in pixel
    
    :return: PCK, predicted, where target_points are the points returned for
             the query
    """
    
    N, _ = query_points.shape
    _, W, _ = feats_query.shape
    
    # find ground truth correspondents in the target image
    # flow has shape (H, W, 2), each being (x, y)
    target_points = flow[query_points[:, 1], query_points[:, 0]].astype(int)
    
    # convert features maps and points into 1D version
    [feats_query, feats_target] = [convert2DTo1D(f) for f in [feats_query, feats_target]]
    query_points_indices = pixel_to_index(query_points, W)
    target_points_indices = pixel_to_index(target_points, W)
    
    # find nearest neighbor in the targe image
    predicted_points = np.array(nn_match(feats_query[query_points_indices], feats_target))
    
    
    # convert from 1D to 2D, shape (N, 2)
    predicted_points = index_to_pixel(predicted_points, W)
    
    
    # find correct predictions
    dist = np.linalg.norm(target_points - predicted_points, axis=1)
    correct = dist <= threshold
    
    # return PCK and predicted points
    return correct.mean(), predicted_points
    
    
def find_correspondence(query_shape, target_shape, H):
    """
    Find matchability and flow map (correspondences) between two images
    
    :param query_shape: shape of the query image, (H, W)
    :param target_shape: shape of the target image, (H0, W0)
    :return: flow, matchability, where
             flow: shape (H, W, 2), where each element is (x, y) location
             matchability: shape (H, W)
    """
    h_query, w_query = query_shape
    h_target, w_target = target_shape
    
    # generate the indexing grid
    # shape (2, H, W)
    yx_query = np.mgrid[:h_query, :w_query]
    # shape (H*W, 2), but in (x, y) form
    xy_query = yx_query[::-1].transpose(1, 2, 0).reshape(-1, 2)
    # use cv2 to apply homography to all locations, result shape (H*W, 1, 2)
    xy_target = cv2.perspectiveTransform(xy_query[:, None, :].astype(float), H)
    # reshape to final result
    flow = xy_target.reshape(h_query, w_query, 2)
    # find matchability map
    matchability = (
            (flow[:, :, 0] >= 0) & (flow[:, :, 0] < w_target) &
            (flow[:, :, 1] >= 0) & (flow[:, :, 1] < h_target))
    
    return flow, matchability
    
    
def convert2DTo1D(feature_map):
    """
    Convert a feature map from 2D to flatten 1D version
    
    :param feature_map: shape (H, W, C)
    :return: (H * W, C)
    """
    H, W, C = feature_map.shape
    return feature_map.reshape(-1, C)
    
def pixel_to_index(xy, width):
    """
    Consistently convert pixel location to index.
    
    :param xy: xy locations, shape (N, 2)
    :param width: image width
    :return: indices into the flattened image, shape (N, )
    """
    
    return xy[:, 1] * width + xy[:, 0]
    
    
def index_to_pixel(index, width):
    """
    Consistently convert pixel location to index.
    
    :param: indices into the flattened image
    :param width: image width
    :return: xy array of shape (N, 2).
    """
    
    y = index // width
    x = index % width
    
    return np.stack([x, y], axis=1)

def nn_match(descs1, descs2):
    """
    Perform nearest neighbor match, using descriptors.
    
    This function uses pyflann
    
    :param descs1: descriptors from image 1, (N1, D)
    :param descs2: descriptors from image 2, (N2, D)
    :return indices: indices into keypoints from image 2, (N1, D)
    """
    # diff = descs1[:, None, :] - descs2[None, :, :]
    # diff = np.linalg.norm(diff, ord=2, axis=2)
    # indices = np.argmin(diff, axis=1)
    
    # flann = cv2.FlannBasedMatcher_create()
    # matches = flann.match(descs1.astype(np.float32), descs2.astype(np.float32))
    # indices = [x.trainIdx for x in matches]
    flann = FLANN()
    indices, _ = flann.nn(descs2, descs1, algorithm="kdtree", trees=4)
    
    return indices


    
def generate_img_grid(h, w):
    """
    indexing utility
    :param h, w: image height and weight
    :return: shape (H, W, 2), each being x, y
    """
    
    # (2, H, W)
    grid = np.mgrid[:h, :w]
    grid = grid[::-1].transpose(1, 2, 0)
    return grid

if __name__ == '__main__':
    dataset = HpatchesDataset('data/hpatches-sequences-release')
    model = DaisyWrapper()
    evaluate_pck_single(dataset, model)
    
    
    
    
    
