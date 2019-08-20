import numpy as np
import sys
sys.path.extend([".", ".."])

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn import metrics
from scipy.spatial.distance import cdist

from tools.hpatches_eval.hpatches_dataset import HpatchesDataset
from tools.hpatches_eval.detector_wrappers import MatterportWrapper, SiftWrapper


def evaluate_matching_score_multiple(database, detectors, thresholds, logfile=None):
    """
    Evaluate matching score on a database of image pairs
    
    :param database: next(iter(database)) should return (img0, img1, H), where
                     img0, img1: image of shape (H, W, C)
                     H: homography from img0 to img1
    :param detectors: a list of detectors. Each should have a method 'detect(img
                      ) -> (kpts, descs)
    :param thresholds: a list of correctness thresholds to test on
    :param logfile: if given, the result will be saved to the file.
    :return: a dictionary. The key will be (str(detector), threshold), and the value
             will be the matching score
    """
    
    results = {}
    
    for thres in thresholds:
        print('# Testing on threshold (pixels): {}'.format(thres))
        print()
        for detector in detectors:
            print('## Testing detector "{}":'.format(detector))
            score = evaluate_matching_score(database, detector, thres)
            results[(str(detector), thres)] = score
            print('matching score: ', score)
    
    return results


def evaluate_matching_score(database, detector, threshold):
    """
    Evaluate matching score on a database of image pairs.
    
    :param database: next(iter(database)) should return (img0, img1, H), where
                     img0, img1: image of shape (H, W, C)
                     H: homography from img0 to img1
    :param detector: should have a method 'detect(img) -> (kpts, descs)',
    :param threshold: correctness threshold
    :return: average matching score
    """
    
    pbar = tqdm(total=len(database))
    scores = []
    for i, data in enumerate(database):
        # if i > 20:
        #     break
        img0, img1, H = data
        kpts0, descs0 = detector.detect(img0)
        kpts1, descs1 = detector.detect(img1)
        # kpts0, kpts1, descs0, descs1 = [x[:num_kp] for x in (kpts0, kpts1, descs0, descs1)]
        (h0, w0), (h1, w1) = img0.shape[:2], img1.shape[:2]
        score = compute_matching_score(descs0, kpts0, descs1, kpts1, H, h0, w0, h1, w1, threshold)
        if score:
            scores.append(score)
        pbar.update()
    
        print("{}: {}".format(i, sum(scores) / len(scores)))
        
    return sum(scores) / len(scores)


def evaluateFPR95(database, detector):
    """
    Evaluate precision recall on a database of image pairs.
    
    :param database: next(iter(database)) should return (img0, img1, H), where
                     img0, img1: image of shape (H, W, C)
                     H: homography from img0 to img1
    :param detector: should have a method 'detect(img) -> (kpts, descs)',
    :return: FPR95 in average
    """
    
    pbar = tqdm(total=len(database))
    FPR95_list = []
    for i, data in enumerate(database):
        img0, img1, H = data
        kpts0, descs0 = detector.detect(img0)
        kpts1, descs1 = detector.detect(img1)
        (h0, w0), (h1, w1) = img0.shape[:2], img1.shape[:2]
        _, _, FPR95 = compute_roc_curve(descs0, kpts0, descs1, kpts1, H, h0, w0, h1, w1)
        FPR95_list.append(FPR95)
        pbar.update()
    
    return sum(FPR95_list) / len(FPR95_list)


def compute_roc_curve(feats0, kps0, feats1, kps1, H, h0, w0, h1, w1, thres_kp=4):
    """
    Compute precision-recall curves.
    
    :param feats0, feats1: descriptors for the images, shape (N0, D)
    :param kps0, kps1: keypoints for the image, shape (N0, 2), each being (x, y)
    :param H: homography, shape (3, 3), from image 0 to image 1
    :param h0, h1, w0, w1: heights and weights for the images
    :param thresh: threshold in pixel, for judging keypoint pair correctness
    :return: fpr_list, tpr_list, FPR95
    """
    
    # only keep keypoints that is present in the other image
    kps0, feats0 = keep_valid_keypoints(kps0, feats0, H, h1, w1)
    kps1, feats1 = keep_valid_keypoints(kps1, feats1, np.linalg.inv(H), h0, w0)
    
    # number of valid proposed matchings
    N0, N1 = kps0.shape[0], kps1.shape[0]
    
    # if none of the keypoints are present, return None
    if N0 == 0 or N1 == 0:
        return None
    
    # map keypoints in image 0 to image 1
    kps0_mapped = cv2.perspectiveTransform(kps0.reshape(N0, 1, 2).astype(float), H).reshape(N0, 2)
    
    # compute keypoint pair distances kps0_mapped (N0, 2), kps1 (N1, 2)
    kps_dist = np.linalg.norm(kps0_mapped[:, None, :] - kps1[None, :, :], ord=2, axis=2)
    # close keypoint pairs are deemd correct pairs
    kps_correct = kps_dist <= thres_kp
    
    # compute descriptor pair distances
    # des_dist = np.linalg.norm(feats0[:, None, :] - feats1[None, :, :], ord=2, axis=2)
    des_dist = cdist(feats0, feats1)
    
    fpr_list, tpr_list, _ = roc_curve(kps_correct.flatten(), 1 - des_dist.flatten())
    
    FPR95_index = np.argmin(np.abs(np.array(tpr_list) - 0.95))
    FPR95 = fpr_list[FPR95_index]
    
    return fpr_list, tpr_list, FPR95


def compute_matching_score(feats0, kps0, feats1, kps1, H, h0, w0, h1, w1, thresh, keep=1000):
    """
    Compute matching score given two sets of keypoints and descriptors.
    
    Note: the sent in keypoints should be sorted by confidence!
    
    :param feats0, feats1: descriptors for the images, shape (N0, D)
    :param kps0, kps1: keypoints for the image, shape (N0, 2), each being (x, y)
    :param H: homography, shape (3, 3), from image 0 to image 1
    :param h0, h1, w1, w2: heights and weights for the images
    :param thresh: threshold in pixel
    :param
    
    :return: matchine score (%)
    """
    
    # only keep keypoints that are present in the other image
    kps0, feats0 = keep_valid_keypoints(kps0, feats0, H, h1, w1)
    kps1, feats1 = keep_valid_keypoints(kps1, feats1, np.linalg.inv(H), h0, w0)
    
    # keep a certain number of keypoints
    # kps0, kps1, feats0, feats1 = [x[:keep] for x in (kps0, kps1, feats0, feats1)]
    
    # number of valid proposed matchings
    N0, N1 = kps0.shape[0], kps1.shape[0]

    # if none of the keypoints are present, return None
    if N0 == 0 or N1 == 0:
        return None
    
    # matches points from image 0 to image 1, using NN
    idxs = nn_match(feats0, feats1)  # matched image 1 keypoint indices
    predicted = kps1[idxs]  # matched image 1 keypoints
    
    # ground truth matched location
    gt = cv2.perspectiveTransform(kps0.reshape(N0, 1, 2).astype(np.float), H).reshape(N0, 2)
    correct0 = np.linalg.norm(predicted - gt, 2, axis=1) <= thresh
    
    # 1 to 0
    idxs = nn_match(feats1, feats0)
    predicted = kps0[idxs]
    gt = cv2.perspectiveTransform(kps1.reshape(N1, 1, 2).astype(np.float), np.linalg.inv(H)).reshape(N1, 2)
    correct1 = np.linalg.norm(predicted - gt, 2, axis=1) <= thresh
    
    return (np.sum(correct1) + np.sum(correct0)) / (N1 + N0)


def keep_valid_keypoints(keypoints, descriptors, H, height, width):
    """
    Keep only keypoints that is present in the other image.
    
    :param keypoints: shape (N, 2)
    :param descriptors: shape (N, D)
    :param H: shape (3, 3)
    :param height, weight: target image height and width
    :return (keypoints, descriptors): valid ones
    """
    
    N = keypoints.shape[0]
    mapped = cv2.perspectiveTransform(keypoints.reshape(N, 1, 2).astype(np.float), H).reshape(N, 2)
    indices_valid = (
            (mapped[:, 0] >= 0) & (mapped[:, 0] < width) &
            (mapped[:, 1] >= 0) & (mapped[:, 1] < height)
    )
    
    return keypoints[indices_valid], descriptors[indices_valid]


def nn_match(descs1, descs2):
    """
    Perform nearest neighbor match, using descriptors.
    
    This function uses OpenCV FlannBasedMatcher
    
    :param descs1: descriptors from image 1, (N1, D)
    :param descs2: descriptors from image 2, (N2, D)
    :return indices: indices into keypoints from image 2, (N1, D)
    """
    # diff = descs1[:, None, :] - descs2[None, :, :]
    # diff = np.linalg.norm(diff, ord=2, axis=2)
    # indices = np.argmin(diff, axis=1)
    
    flann = cv2.FlannBasedMatcher_create()
    matches = flann.match(descs1.astype(np.float32), descs2.astype(np.float32))
    indices = [x.trainIdx for x in matches]
    
    return indices


def roc_curve(y_true, y_predict):
    """
    Compute ROC curve
    
    :param y_true: true labels, shape (N,)
    :param y_predict: predicted scores, shape (N,)
    :return: fpr_list, tpr_list, thresholds
    """
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(y_true, y_predict)
    
    return fpr_list, tpr_list, threshold_list


if __name__ == "__main__":
    database = HpatchesDataset(root="./data/HPATCHES")
    detector = MatterportWrapper()
    print("matching score: {}".format(evaluate_matching_score(database, detector, threshold=3)))
