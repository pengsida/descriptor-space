import pickle
import os
import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F
import scipy.signal as signal
from matplotlib import cm
import matplotlib.pyplot as plt

from lib.utils.hard_mining.hard_example_mining_layer import hard_example_mining_layer
from lib.utils.logger import logger


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def convert_to_rgb(img_raw):
    if len(img_raw.shape) == 2:
        img_raw = np.repeat(img_raw[:, :, None], 3, axis=2)
    if img_raw.shape[2] > 3:
        img_raw = img_raw[:, :, :3]
    return img_raw


def denormalize(img):
    img *= [0.229, 0.224, 0.225]
    img += [0.485, 0.456, 0.406]
    img *= 255
    return img


def compute_homography_discrete(h,w,scale=True,rotation=True,translation=True,
                                scale_range=(2.0,1.0,0.5,),max_scale_disturb=0.0,
                                max_angle=np.pi/6, translation_overflow=0.):
    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    pts2 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    scale_offset = 0

    if scale:
        scale_ratio=np.random.choice(scale_range,1)*(1+np.random.uniform(-max_scale_disturb,max_scale_disturb))
        pts2=(pts2-0.5)*scale_ratio+0.5
        scale_offset = math.log2(scale_ratio)

    if translation:
        t_min, t_max = np.min(pts2, axis=0) + translation_overflow, \
                       np.min(1 - pts2, axis=0) + translation_overflow
        if -t_min[0]>=t_max[0]: tx=0
        else: tx=np.random.uniform(-t_min[0],t_max[0])
        if -t_min[1]>=t_max[1]: ty=0
        else: ty=np.random.uniform(-t_min[1],t_max[1])
        pts2[:,0]+=tx
        pts2[:,1]+=ty

    if rotation:
        angle = np.random.uniform(-max_angle, max_angle)
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_m = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],np.float32)
        pts2 = np.matmul((pts2 - center), rot_m.transpose()) + center

    pts2+=np.random.uniform(0,0.05,pts2.shape)

    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape
    return cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32)), scale_offset


def detect_dog_keypoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    if len(img.shape)==3:
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kps=sift.detect(img,None)
    kps_np=np.asarray([kp.pt for kp in kps])
    return np.round(kps_np).astype(np.int32)


def round_coordinates(coord,h,w):
    coord=np.round(coord).astype(np.int32)
    coord[coord[:,0]<0,0]=0
    coord[coord[:,0]>=w,0]=w-1
    coord[coord[:,1]<0,1]=0
    coord[coord[:,1]>=h,1]=h-1
    return coord


def draw_corspd(img0, pix_pos0, img1, pix_pos1):
    num = 10
    inds = np.random.choice(range(len(pix_pos0)), num, replace=False)
    pix_pos0 = pix_pos0[inds]
    pix_pos1 = pix_pos1[inds]

    kps0 = [cv2.KeyPoint(pix[0], pix[1], 1) for pix in pix_pos0]
    kps1 = [cv2.KeyPoint(pix[0], pix[1], 1) for pix in pix_pos1]
    matches = [cv2.DMatch(i, i, 0) for i in range(num)]
    img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)

    return img


def draw_kps(img0, pix_pos0, img1, pix_pos1):
    kps0 = [cv2.KeyPoint(pix[0], pix[1], 1) for pix in pix_pos0]
    img0 = cv2.drawKeypoints(img0, kps0, None)
    kps1 = [cv2.KeyPoint(pix[0], pix[1], 1) for pix in pix_pos1]
    img1 = cv2.drawKeypoints(img1, kps1, None)
    img = np.concatenate([img0, img1], axis=1)
    return img


def draw_triplet(img0, pix_pos0, img1, pix_pos1, img2, pix_pos2):
    img_top = draw_corspd(img0, pix_pos0, img1, pix_pos1)
    img_bottom = draw_corspd(img0, pix_pos0, img2, pix_pos2)
    return np.concatenate([img_top, img_bottom], axis=0)


def random_crop(img, size):
    h, w = img.shape[:2]
    sz_h, sz_w = size

    if h > sz_h:
        i = np.random.randint(0, h-sz_h)
    else:
        i = 0

    if w > sz_w:
        j = np.random.randint(0, w-sz_w)
    else:
        j = 0

    img = img[i:i+sz_h, j:j+sz_w]
    h, w = img.shape[:2]

    if h != sz_h or w != sz_w:
        img = cv2.resize(img, (sz_w, sz_h), interpolation=cv2.INTER_LINEAR)

    return img


def resize_max_length(img, length):
    h, w = img.shape[:2]
    ratio = length / max(h, w)
    h, w = int(round(h * ratio)), int(round(w * ratio))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def sift_detector(img):
    sift = cv2.xfeatures2d.SIFT_create()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kps = sift.detect(img, None)
    kps_np = np.asarray([kp.pt for kp in kps])
    return np.round(kps_np).astype(np.int32)


def sample_descriptor(descs, kps, images):
    """
    descs: [B, D, H', W']
    kps: [B, N, 2], kps are the original pixels of images
    images: [B, 3, H, W]
    :return descs [B, N, D]
    """
    h, w = images.shape[2:]
    with torch.no_grad():
        kps = kps.clone().detach()
        kps[:, :, 0] = (kps[:, :, 0] / (float(w)/2.)) - 1.
        kps[:, :, 1] = (kps[:, :, 1] / (float(h)/2.)) - 1.
        kps = kps.unsqueeze(dim=1)
    descs = F.grid_sample(descs, kps)
    descs = descs[:, :, 0, :].permute(0, 2, 1)
    return F.normalize(descs, p=2, dim=2)


def sample_scale(scales, kps, images):
    """
    scales: [B, 1, H', W']
    kps: [B, N, 2], kps are the original pixels of images
    images: [B, 3, H, W]
    :return scales [B, N]
    """
    h, w = images.shape[2:]
    with torch.no_grad():
        kps = kps.clone().detach()
        kps[:, :, 0] = (kps[:, :, 0] / (float(w)/2.)) - 1.
        kps[:, :, 1] = (kps[:, :, 1] / (float(h)/2.)) - 1.
        kps = kps.unsqueeze(dim=1)
    scales = F.grid_sample(scales, kps)
    return scales[:, 0, 0]


def hard_negative_mining(desc_seeds, desc_maps, kps, images, thresh=16, interval=16):
    """
    The mined locations should be thresh pixels away from the kps.
    To reduce the computational cost, we sample the negative locations every interval pixels.
    desc_seeds: [B, N, D]
    desc_maps: [B, D, H', W']
    kps: [B, N, 2]
    images: [B, 3, H, W]
    :return descs [B, N, D]
    """
    with torch.no_grad():
        # rescale the kps to the size of desc_map
        ratio_h = desc_maps.shape[2] / images.shape[2]
        ratio_w = desc_maps.shape[3] / images.shape[3]
        ratio = torch.tensor([ratio_w, ratio_h]).cuda()
        kps = kps.clone().detach()
        kps *= ratio

        # hard negative mining
        neg_kps = hard_example_mining_layer(desc_maps, desc_seeds, kps, thresh, interval).float()  # [B, N, 3]
        neg_kps = neg_kps[:, :, 1:]
        neg_kps /= ratio

    logger.update(kps2=neg_kps[0])
    descs = sample_descriptor(desc_maps, neg_kps, images)
    return descs


def draw_corspd_region(img0, img1, H):
    """
    img0: [H_0, W_0, 3]
    img1: [H_1, W_1, 3]
    """
    h1, w1 = img1.shape[:2]
    pts1 = np.array(
        [[0, 0],
         [0, h1],
         [w1, h1],
         [w1, 0]]
    ).astype(np.float32)
    pts0 = cv2.perspectiveTransform(np.reshape(pts1, [1, -1, 2]), np.linalg.inv(H))[0]

    # draw the corresponding region on the image
    pts0 = pts0.astype(np.int32)
    img0 = cv2.polylines(img0.copy(), [pts0.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    pts1 = pts1.astype(np.int32)
    img1 = cv2.polylines(img1.copy(), [pts1.reshape([-1, 2])], True, (255, 0, 0), thickness=5)

    return np.concatenate([img0, img1], axis=1)


def warp_img_by_pts(img, pts2):
    """
    img: [H, W, 3]
    pts2: control points, [left_top, left_bottom, right_bottom, right_top]
    """
    pts1 = np.array(
        [[0., 0.],
         [0., 1.],
         [1., 1.],
         [1., 0.]]
    )

    # rescale the control points
    h, w = img.shape[:2]
    shape = np.array([[w, h]]).astype(np.float32)
    pts1 *= shape
    pts2 *= shape

    H = cv2.getPerspectiveTransform(pts2.astype(np.float32), pts1.astype(np.float32))
    img = cv2.warpPerspective(img.copy(), H, (w, h), flags=cv2.INTER_LINEAR)

    return img


def homo_mm(H, x, y):
    [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = H
    C = m31 * x + m32 * y + m33
    x_prime = m11 * x + m12 * y + m13
    y_prime = m21 * x + m22 * y + m23
    return x_prime, y_prime, C


def compute_scale(H, h, w):
    """
    compute the pixel scale changes after homography transformation
    H: homography matrix, [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]]
    """
    [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = H
    yx = np.mgrid[:h, :w]
    x, y, C = homo_mm(H, yx[1], yx[0])
    # J: [2, 2, h, w]
    J = np.array([[m11 * C - m31 * x, m12 * C - m32 * x],
                 [m21 * C - m31 * y, m22 * C - m32 * y]])

    J /= C ** 2
    J = J.transpose(2, 3, 0, 1)
    # scale = np.sqrt(np.linalg.det(J)).astype(np.float32)
    scale = np.sqrt(np.abs(np.linalg.det(J))).astype(np.float32)
    return scale


def compute_scale_xy(H, h, w):
    """
    compute the pixel scale changes after homography transformation
    H: homography matrix, [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]]
    """
    yx = np.mgrid[:h+1, :w+1]
    X = yx[1] * H[0, 0] + yx[0] * H[0, 1] + H[0, 2]
    Y = yx[1] * H[1, 0] + yx[0] * H[1, 1] + H[1, 2]
    C = yx[1] * H[2, 0] + yx[0] * H[2, 1] + H[2, 2]
    X /= C
    Y /= C
    # the pixel size in x direction
    delta_x = (X[:, 1:] - X[:, :-1])[:-1]
    # the pixel size in y direction
    delta_y = (Y[1:] - Y[:-1])[:, :-1]
    scale = (delta_x + delta_y) / 2.

    # A = H[:2, :2]
    # m31, m32 = H[2, :2]
    # yx = np.mgrid[:h, :w]
    # C = m31 * yx[1] + m32 * yx[0] + 1.
    # scale = (np.sqrt(np.linalg.det(A) / C)).astype(np.float32)
    return scale


class SampleTrainingTarget(object):
    @staticmethod
    def get_homography_correspondence(h, w, H):
        coords = [np.expand_dims(item, 2) for item in np.meshgrid(np.arange(w), np.arange(h))]
        coords = np.concatenate(coords, 2).astype(np.float32)
        coords_target = cv2.perspectiveTransform(np.reshape(coords, [1, -1, 2]), H.astype(np.float32))
        coords_target = np.reshape(coords_target, [h, w, 2])

        source_mask = np.logical_and(np.logical_and(0 <= coords_target[:, :, 0], coords_target[:, :, 0] < w - 0),
                                     np.logical_and(0 <= coords_target[:, :, 1], coords_target[:, :, 1] < h - 0))
        coords_target[np.logical_not(source_mask)] = 0

        return coords_target, source_mask

    @staticmethod
    def uniform_sample_correspondence(pix_pos, msk):
        hs_ws = np.argwhere(msk)
        sample_num = 3000
        if len(hs_ws) >= sample_num:
            inds = np.random.choice(np.arange(len(hs_ws)), sample_num, replace=False)
        else:
            inds = np.arange(len(hs_ws))
        hs_ws = hs_ws[inds]
        hs = hs_ws[:, 0]
        ws = hs_ws[:, 1]
        pos_num = len(hs)

        if pos_num >= sample_num:
            idxs = np.arange(pos_num)
            np.random.shuffle(idxs)
            idxs = idxs[:sample_num]
        else:
            idxs = np.arange(pos_num)
            idxs = np.append(idxs, np.random.choice(idxs, sample_num - pos_num))

        pix_pos0 = np.concatenate([ws[idxs][:, None], hs[idxs][:, None]], 1)  # sn,2
        pix_pos1 = pix_pos[pix_pos0[:, 1], pix_pos0[:, 0]]

        return pix_pos0, pix_pos1

    @staticmethod
    def sample_ground_truth(h, w, H):
        pix_pos, msk = SampleTrainingTarget.get_homography_correspondence(h, w, H)
        pix_pos0, pix_pos1 = SampleTrainingTarget.uniform_sample_correspondence(pix_pos, msk)
        return pix_pos0, pix_pos1

    @staticmethod
    def sample_negative(h, w, pix_pos):
        """
        pix_pos: [N, 2]
        """
        pix_pos1 = pix_pos.copy()
        pix_x = np.random.randint(5, w-5, len(pix_pos1))
        pix_y = np.random.randint(5, h-5, len(pix_pos1))
        pix_pos1[:, 0] = (pix_pos1[:, 0] + pix_x) % w
        pix_pos1[:, 1] = (pix_pos1[:, 1] + pix_y) % h
        return pix_pos1

    @staticmethod
    def sample(h0, w0, h1, w1, H):
        """
        H: homography matrix, [3, 3]
        """
        # compute kps
        pix_pos0, pix_pos1 = SampleTrainingTarget.sample_ground_truth(h0, w0, H)
        pix_pos2 = SampleTrainingTarget.sample_negative(h1, w1, pix_pos1)

        # compute relative scale
        scale0 = compute_scale(H, h0, w0)
        scale0 = (1. / scale0).reshape(1, h0, w0)
        scale1 = compute_scale(np.linalg.inv(H), h1, w1)
        scale1 = (1. / scale1).reshape(1, h1, w1)

        return {
            'kps0': pix_pos0,
            'kps1': pix_pos1,
            'kps2': pix_pos2,
            'scale0': scale0,
            'scale1': scale1,
            'H': H
        }

    @staticmethod
    def sample_torch(image0, image1, H):
        """
        image0: [3, H0, W0]
        image1: [3, H1, W1]
        """
        if isinstance(H, torch.Tensor):
            H = H.detach().cpu().numpy()

        h0, w0 = image0.shape[1:]
        h1, w1 = image1.shape[1:]
        target = SampleTrainingTarget.sample(h0, w0, h1, w1, H)
        return {k: torch.tensor(v).float() for k, v in target.items()}


def compute_local_contrast(img, N=1):
    """
    img: [H, W, 3]
    """
    img = img[:, :, 0]
    im = np.array(img, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)

    kernel = np.ones((2*N+1, 2*N+1))
    s = signal.convolve2d(im, kernel, mode="same")
    s2 = signal.convolve2d(im2, kernel, mode="same")
    ns = signal.convolve2d(ones, kernel, mode="same")

    return np.sqrt((s2 - s**2 / ns) / ns)


def draw_scale(scale):
    """
    scale: [H, W]
    """
    fig, ax = plt.subplots(1, figsize=(2.4, 2.4))
    mappable = ax.imshow(scale)
    fig.colorbar(mappable, ax=ax)
    mappable.set_clim(0, 2)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    buf = np.roll(buf, 3, axis=2)[:, :, :3]
    plt.close()
    return buf


def get_homography_correspondence(h, w, H):
    coords = [np.expand_dims(item, 2) for item in np.meshgrid(np.arange(w), np.arange(h))]
    coords = np.concatenate(coords, 2).astype(np.float32)
    coords_target = cv2.perspectiveTransform(np.reshape(coords, [1, -1, 2]), H.astype(np.float32))
    coords_target = np.reshape(coords_target, [h, w, 2])

    source_mask = np.logical_and(np.logical_and(0 <= coords_target[:, :, 0], coords_target[:, :, 0] < w - 0),
                                 np.logical_and(0 <= coords_target[:, :, 1], coords_target[:, :, 1] < h - 0))
    coords_target[np.logical_not(source_mask)] = 0

    return coords_target, source_mask


def compute_cost_volume(left, right):
    """
    left correlates with right, resulting in a cost volume with size [B, H1*W1, H0, W0]

    left: [B, D, H0, W0]
    right: [B, D, H1, W1]
    """
    b, d, h0, w0 = left.shape
    left = left.view(b, d, h0*w0).permute(0, 2, 1)
    h1, w1 = right.shape[2:]
    right = right.view(b, d, h1*w1)

    # [b, h0*w0, h1*w1]
    cost_volume = torch.matmul(left, right)
    # [b, h1*w1, h0, w0]
    cost_volume = cost_volume.permute(0, 2, 1).view(b, h1*w1, h0, w0)
    return cost_volume


def draw_scale_cls(scale):
    """
    scale: [H, W]
    """
    scale[scale > 1.5] = 2
    scale[scale < 0.75] = 0
    scale[(scale >= 0.75) * (scale <= 1.5)] = 1
    r = scale.copy()
    g = scale.copy()
    b = scale.copy()
    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70]
    ]
    for l in range(3):
        inds = (scale == l)
        r[inds] = colors[l][0]
        g[inds] = colors[l][1]
        b[inds] = colors[l][2]
    cls = np.stack([r, g, b], axis=2).astype(np.uint8)
    return cls


def sample_pyramid_descriptor(pyramid_descriptor, kps, image):
    """
    pyramid_descriptor: [P, D, H', W']
    kps: [N, 2], the pixels of original image
    image: [3, H, W]
    """
    p = pyramid_descriptor.shape[0]
    descs = []
    for i in range(p):
        desc = sample_descriptor(pyramid_descriptor[i].unsqueeze(0), kps.unsqueeze(0), image.unsqueeze(0))[0]
        descs.append(desc)

    # distance_matrix = np.zeros(shape=[p, p])
    # for i in range(p):
    #     for j in range(p):
    #         distance_matrix[i][j] = torch.norm(descs[i] - descs[j]).detach().cpu().numpy()
    #
    # li_distance_matrix = np.zeros(shape=[p//2])
    # for i in range(1, p//2+1):
    #     desc = 0.5 * descs[5 - i] + 0.5 * descs[5 + i]
    #     li_distance_matrix[i-1] = torch.norm(descs[5] - desc)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(distance_matrix)
    # plt.colorbar()
    # plt.show()

    return torch.stack(descs, dim=1)
