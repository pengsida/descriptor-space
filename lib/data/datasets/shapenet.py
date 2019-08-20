import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import glob
import time
import matplotlib.pyplot as plt

from lib.utils.base import read_pickle, save_pickle
from lib.utils.registry import Registry


def haversine(baseline, viewpoints):
    """
    baseline: [2,] (in decimal degrees)
    viewpoints: [N, 2] (in decimal degrees)
    Calculate the great circle distances of viewpoints from baseline
    reference: https://en.wikipedia.org/wiki/Great-circle_distance
    """
    # convert decimal degrees to radians
    baseline = np.deg2rad(baseline)
    viewpoints = np.deg2rad(viewpoints)
    # haversine formula
    lat1 = baseline[1]
    lat2 = viewpoints[:, 1]
    dlon = viewpoints[:, 0] - baseline[0]
    dlat = viewpoints[:, 1] - baseline[1]
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c


def get_scale(cfg):
    stage = int(cfg.MODEL.BACKBONE.STOP_DOWNSAMPLING[1:])
    scale = 2**(stage-1)
    return scale


class ShapeNetDB(object):
    intrinsic_matrix = {
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }

    @staticmethod
    def read_depth(dmap_path):
        return np.array(Image.open(dmap_path)) / 65535 * 10

    @staticmethod
    def read_pose(pose_path):
        """
        pose stored in pkl
        :param pose_path: *.pkl
        :return: [R|t] in shape(4,4)
        """
        pose = read_pickle(pose_path)['RT']
        return pose

    @staticmethod
    def read_data(rgb_path, dmap_path, pose_path):
        dmap = ShapeNetDB.read_depth(dmap_path)
        pose = ShapeNetDB.read_pose(pose_path)
        data_return = [np.array(Image.open(rgb_path))[..., :3],
                       dmap,
                       pose]
        return data_return

    @staticmethod
    def compute_correspondence(dmap1, dmap2, pose1, pose2, scale=None):
        """ compute correspondence between image1 and image2
        1. get point cloud of image1 and project it to image2
        2. establish dense correspondences between image1 and image2
        """
        # 1. get point cloud of image1 and project it to image2
        height, width = dmap1.shape
        row_col1 = np.argwhere(dmap1 != 10)
        xy = row_col1[:, [1, 0]].astype(np.float32)
        depth = dmap1[row_col1[:, 0], row_col1[:, 1]]
        depth = depth[:, np.newaxis]
        xy *= depth
        xyz = np.concatenate([xy, depth], axis=1)
        camera_points1 = np.dot(xyz, np.linalg.inv(ShapeNetDB.intrinsic_matrix['blender']).T)
        world_points = np.dot(camera_points1 - pose1[:, 3], np.linalg.inv(pose1[:, :3]).T)

        camera_points2 = np.dot(world_points, pose2[:, :3].T) + pose2[:, 3]
        xyz = np.dot(camera_points2, ShapeNetDB.intrinsic_matrix['blender'].T)
        xyz[:, :2] /= xyz[:, 2:]
        row_col2 = np.round(xyz[:, [1, 0]]).astype(np.int32)
        row_col2[:, 0] = np.clip(row_col2[:, 0], a_min=0, a_max=height-1)
        row_col2[:, 1] = np.clip(row_col2[:, 1], a_min=0, a_max=width-1)
        depth = dmap2[row_col2[:, 0], row_col2[:, 1]]

        # 2. establish dense correspondences between image1 and image2
        chosen = (xyz[:, 2] < (depth + 1e-5))
        row_col1 = row_col1[chosen]
        row_col2 = np.round(xyz[:, [1, 0]][chosen]).astype(np.int32)

        if scale is not None:
            height = height // scale
            width = width // scale

            row_col1 = np.round(row_col1 / scale).astype(np.int32)
            row_col1, inds = np.unique(row_col1, return_index=True, axis=0)
            row_col2 = np.round(row_col2 / scale).astype(np.int32)
            row_col2 = row_col2[inds]

            row_col1[:, 0] = np.clip(row_col1[:, 0], 0, height-1)
            row_col1[:, 1] = np.clip(row_col1[:, 1], 0, width-1)
            row_col2[:, 0] = np.clip(row_col2[:, 0], 0, height-1)
            row_col2[:, 1] = np.clip(row_col2[:, 1], 0, width-1)

        return row_col1, row_col2

    @staticmethod
    def compute_flow(dmap1, dmap2, pose1, pose2, scale=None):
        """ compute flow from image1 to image2
        1. establish dense correspondences between image1 and image2
        2. compute dense flow between image1 and image2
        """
        # 1. establish dense correspondences between image1 and image2
        row_col1, row_col2 = ShapeNetDB.compute_correspondence(dmap1, dmap2, pose1, pose2, scale)

        # 2. compute dense flow between image1 and image2
        height, width = dmap1.shape
        if scale is None:
            flow = (row_col2 - row_col1).astype(np.int32)
            flow_map = np.ones([height, width, 2], dtype=np.int32) * 65535
            flow_map[row_col1[:, 0], row_col1[:, 1]] = flow
        else:
            height = height // scale
            width = width // scale
            flow_map = np.ones([height, width, 2], dtype=np.int32) * 65535
            flow_map[row_col1[:, 0], row_col1[:, 1]] = row_col2 - row_col1

        return flow_map

    @staticmethod
    def validate_flow(img1, img2, flow):
        row_col1 = np.argwhere(flow[..., 0] != 65535)
        flow = flow[row_col1[:, 0], row_col1[:, 1]]

        height, width = img1.shape[:2]
        row_col2 = row_col1 + flow
        row_col2[:, 0] = np.clip(row_col2[:, 0], a_min=0, a_max=height-1)
        row_col2[:, 1] = np.clip(row_col2[:, 1], a_min=0, a_max=width-1)

        pixel_value = img2[row_col2[:, 0], row_col2[:, 1]]
        img = np.zeros_like(img2)
        img[row_col1[:, 0], row_col1[:, 1]] = pixel_value

        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img1)
        ax2.imshow(img2)
        ax3.imshow(img)
        plt.show()

    @staticmethod
    def validate_corr(img1, img2, row_col1, row_col2):
        """
        img1: [H, W, 3]
        img2: [H, W, 3]
        row_col1: [N, 2]
        row_col2: [N, 2]
        """
        N, _ = row_col1.shape
        inds = np.linspace(0, N-1, 10, dtype=np.int32)
        row_col1 = row_col1[inds]
        row_col2 = row_col2[inds]

        W = img1.shape[1]
        image = np.concatenate([img1, img2], axis=1)
        row_col2[:, 1] += W

        for rc1, rc2 in zip(row_col1, row_col2):
            plt.plot([rc1[1], rc2[1]], [rc1[0], rc2[0]])
        plt.imshow(image)
        plt.show()

    @staticmethod
    def draw_grad(img1, img2, corr_map1, corr_map2):
        """
        img1: [H, W, 3]
        img2: [H, W, 3]
        corr_map1: [H, W, 2]
        corr_map2: [H, W, 2]
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(img1)
        ax2.imshow(img2)
        ax3.imshow(corr_map1[:, :, 0])
        ax4.imshow(corr_map2[:, :, 0])
        plt.show()

    @staticmethod
    def get_ids(ann_file):
        print("loading annotations into memory..")
        tic = time.time()
        ids = make_ann_file(ann_file)
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        return ids


class ContrastiveDB(object):
    """
    the target of contrastive: the correspondences are close and the non-correspondences are distant
    """
    @staticmethod
    def get_corr(dmap1, dmap2, pose1, pose2, scale):
        row_col1, row_col2 = ShapeNetDB.compute_correspondence(dmap1, dmap2, pose1, pose2, scale)
        correspondence1 = np.concatenate([row_col1, row_col2], axis=1)
        row_col2, row_col1 = ShapeNetDB.compute_correspondence(dmap2, dmap1, pose2, pose1, scale)
        correspondence2 = np.concatenate([row_col1, row_col2], axis=1)
        correspondence = np.concatenate([correspondence1, correspondence2])
        correspondence = np.unique(correspondence, axis=0)
        row_col1, row_col2 = np.split(correspondence, 2, axis=1)
        return row_col1, row_col2

    @staticmethod
    def get_non_corr(cfg, row_col1, fg_row_col2, one_to_two, two_to_one, height, width):
        inds = np.random.randint(0, len(fg_row_col2), len(row_col1))
        non_row_col2 = fg_row_col2[inds]
        non_corr_map = np.ones([height, width, 2], dtype=np.int32) * 65535
        non_corr_map[row_col1[:, 0], row_col1[:, 1]] = non_row_col2
        collide_map = (np.linalg.norm(one_to_two - non_corr_map, axis=2) < cfg.DATASET.NEIGHBORHOOD)
        non_row_col1 = np.argwhere(~collide_map)
        non_row_col2 = non_corr_map[non_row_col1[:, 0], non_row_col1[:, 1]]

        non_corr_map = np.ones([height, width, 2], dtype=np.int32) * 65535
        non_corr_map[non_row_col2[:, 0], non_row_col2[:, 1]] = non_row_col1
        collide_map = (np.linalg.norm(two_to_one - non_corr_map, axis=2) < cfg.DATASET.NEIGHBORHOOD)
        coherent_map = (~collide_map) * (non_corr_map[:, :, 0] != 65535)
        non_row_col2 = np.argwhere(coherent_map)
        non_row_col1 = non_corr_map[non_row_col2[:, 0], non_row_col2[:, 1]]

        return non_row_col1, non_row_col2

    @staticmethod
    def get_fg_bg(cfg, row_col1, bg_row_col2, one_to_two, height, width):
        inds = np.random.randint(0, len(bg_row_col2), len(row_col1))
        non_row_col2 = bg_row_col2[inds]

        non_corr_map = np.ones([height, width, 2], dtype=np.int32) * 65535
        non_corr_map[row_col1[:, 0], row_col1[:, 1]] = non_row_col2
        collide_map = (np.linalg.norm(one_to_two - non_corr_map, axis=2) < cfg.DATASET.NEIGHBORHOOD)
        fg_row_col1 = np.argwhere(~collide_map)
        bg_row_col2 = non_corr_map[fg_row_col1[:, 0], fg_row_col1[:, 1]]

        return fg_row_col1, bg_row_col2

    @staticmethod
    def get_target(cfg, img1, img2, dmap1, dmap2, pose1, pose2, scale):
        # 1. get correspondences
        row_col1, row_col2 = ContrastiveDB.get_corr(dmap1, dmap2, pose1, pose2, scale)

        if cfg.DATASET.SHOW_FLOW:
            collect_flow(cfg, img1, img2, dmap1, dmap2, pose1, pose2)

        # 2. get non-correspondences
        height, width = dmap1.shape
        height = height // scale
        width = width // scale

        one_to_two = np.ones([height, width, 2], dtype=np.int32) * 65535
        one_to_two[row_col1[:, 0], row_col1[:, 1]] = row_col2
        two_to_one = np.ones([height, width, 2], dtype=np.int32) * 65535
        two_to_one[row_col2[:, 0], row_col2[:, 1]] = row_col1

        mask2 = Image.fromarray((dmap2 != 10).astype(np.int32))
        mask2 = np.array(mask2.resize((width, height)))
        fg_row_col2 = np.argwhere(mask2 != 0)
        bg_row_col2 = np.argwhere(mask2 == 0)

        non_row_col1, non_row_col2 = ContrastiveDB.get_non_corr(
            cfg,
            row_col1, fg_row_col2,
            one_to_two, two_to_one,
            height, width,
        )
        fg_row_col1, bg_row_col2 = ContrastiveDB.get_fg_bg(cfg, row_col1, bg_row_col2, one_to_two, height, width)

        corr_map = np.ones([height, width, 2], dtype=np.int32) * 65535
        corr_map[row_col1[:, 0], row_col1[:, 1]] = row_col2
        non_corr_map = np.ones([height, width, 2], dtype=np.int32) * 65535
        non_corr_map[non_row_col1[:, 0], non_row_col1[:, 1]] = non_row_col2
        fg_bg_map = np.ones([height, width, 2], dtype=np.int32) * 65535
        fg_bg_map[fg_row_col1[:, 0], fg_row_col1[:, 1]] = bg_row_col2

        corr_map = torch.from_numpy(corr_map).float()
        non_corr_map = torch.from_numpy(non_corr_map).float()
        fg_bg_map = torch.from_numpy(fg_bg_map).float()
        target = dict(corr=corr_map, non_corr=non_corr_map, fg_bg=fg_bg_map)

        return target


TARGETS = Registry()


@TARGETS.register("FLOW")
def collect_flow(cfg, img1, img2, dmap1, dmap2, pose1, pose2, **kwargs):
    """
    the target of flow: flow and matchability
    """
    scale = get_scale(cfg)
    flow = ShapeNetDB.compute_flow(dmap1, dmap2, pose1, pose2, scale)

    if cfg.DATASET.SHOW_FLOW:
        height, width = dmap1.shape
        height = height // scale
        width = width // scale
        img1 = np.array(Image.fromarray(img1).resize([width, height]))
        img2 = np.array(Image.fromarray(img2).resize([width, height]))
        ShapeNetDB.validate_flow(img1, img2, flow)

    flow = torch.from_numpy(flow).permute(2, 0, 1).float()
    matchability = (flow[0] != 65535).unsqueeze(0).float()
    target = dict(flow=flow, matchability=matchability)

    return target


@TARGETS.register("FEATURE")
def collect_contrastive(cfg, img1, img2, dmap1, dmap2, pose1, pose2, **kwargs):
    """
    the target of contrastive: the correspondences are close and the non-correspondences are distant
    """
    scale = get_scale(cfg)
    target = ContrastiveDB.get_target(cfg, img1, img2, dmap1, dmap2, pose1, pose2, scale)

    if cfg.MODEL.META_ARCHITECTURE == "AdaptiveOneShot":
        target_2s = ContrastiveDB.get_target(cfg, img1, img2, dmap1, dmap2, pose1, pose2, scale*2)
        target_2s = {k+"_2s": v for k, v in target_2s.items()}
        target.update(target_2s)

    if cfg.MODEL.META_ARCHITECTURE == "GroundTruthOneShot":
        row_col1, row_col2 = ContrastiveDB.get_corr(dmap1, dmap2, pose1, pose2, scale)
        height, width = dmap1.shape
        height = height // scale
        width = width // scale

        if cfg.DATASET.SHOW_CORR:
            img1 = np.array(Image.fromarray(img1).resize([width, height]))
            img2 = np.array(Image.fromarray(img2).resize([width, height]))
            ShapeNetDB.validate_corr(img1, img2, row_col1, row_col2)

        corr_map1 = np.zeros([height, width, 2])
        corr_map1[row_col1[:, 0], row_col1[:, 1]] = row_col2
        corr_map2 = np.zeros([height, width, 2])
        corr_map2[row_col2[:, 0], row_col2[:, 1]] = row_col1

        if cfg.DATASET.SHOW_GRAD:
            img1 = np.array(Image.fromarray(img1).resize([width, height]))
            img2 = np.array(Image.fromarray(img2).resize([width, height]))
            ShapeNetDB.draw_grad(img1, img2, corr_map1, corr_map2)

        corr_map = np.concatenate([corr_map1, corr_map2], axis=2)
        corr_map = torch.from_numpy(corr_map).permute(2, 0, 1).float()
        target["corr_map"] = corr_map

    if cfg.MODEL.META_ARCHITECTURE == "PseudoOneShot":
        idx = kwargs["idx"]
        folder = kwargs["ann_file"].replace(".pkl", "")
        corr_map_path = os.path.join(folder, "corr_map", "{}.pth".format(idx))
        corr_map1, corr_map2 = torch.load(corr_map_path)

        if cfg.DATASET.SHOW_GRAD:
            _, height, width = corr_map1.shape
            img1 = np.array(Image.fromarray(img1).resize([width, height]))
            img2 = np.array(Image.fromarray(img2).resize([width, height]))
            ShapeNetDB.draw_grad(img1, img2, corr_map1.permute(1, 2, 0), corr_map2.permute(1, 2, 0))

        corr_map = torch.cat([corr_map1, corr_map2], dim=0).float()
        target["corr_map"] = corr_map

    return target


@TARGETS.register("WEIGHT")
def collect_contrastive(cfg, img1, img2, dmap1, dmap2, pose1, pose2, **kwargs):
    """
    the target of contrastive: the correspondences are close and the non-correspondences are distant
    """
    scale = get_scale(cfg)
    target = ContrastiveDB.get_target(cfg, img1, img2, dmap1, dmap2, pose1, pose2, scale)
    idx = kwargs["idx"]
    folder = kwargs["ann_file"].replace(".pkl", "")
    corr_map_path = os.path.join(folder, "corr_map", "{}.pth".format(idx))
    corr_map1, _ = torch.load(corr_map_path)
    target["corr_map"] = corr_map1.float()

    return target


BASELINES = Registry()


@BASELINES.register("ShapeNet_car_small-baseline")
def get_small_baseline(affinity):
    baseline_begin = 1
    baseline_end = 11
    ids = np.arange(affinity.shape[0])
    affinity = affinity[:, baseline_begin:baseline_end]
    return ids, affinity


@BASELINES.register("ShapeNet_car_large-baseline")
def get_large_baseline(affinity):
    baseline_begin = 45
    baseline_end = 55
    ids = np.arange(affinity.shape[0])
    affinity = affinity[:, baseline_begin:baseline_end]
    return ids, affinity


@BASELINES.register("ShapeNet_car_random-baseline")
def get_random_baseline(affinity):
    baseline_begin = 1
    baseline_end = 35
    N = affinity.shape[0]
    K = 10
    inds = np.random.randint(baseline_begin, baseline_end, [N, K])
    ids = np.arange(N)
    affinity = affinity[ids[:, np.newaxis], inds]
    return ids, affinity


@BASELINES.register("ShapeNet_car_mini-small")
def get_mini_dataset(affinity):
    baseline_begin = 1
    K = 10
    baseline_end = baseline_begin + K
    N = 10
    inds = np.arange(baseline_begin, baseline_end)
    ids = np.linspace(0, affinity.shape[0]-1, N, dtype=np.int32)
    affinity = affinity[ids[:, np.newaxis], inds]
    return ids, affinity


@BASELINES.register("ShapeNet_car_mini-large")
def get_mini_dataset(affinity):
    baseline_begin = 45
    K = 10
    baseline_end = baseline_begin + K
    N = 10
    inds = np.arange(baseline_begin, baseline_end)
    ids = np.linspace(0, affinity.shape[0]-1, N, dtype=np.int32)
    affinity = affinity[ids[:, np.newaxis], inds]
    return ids, affinity


def make_ann_file(ann_file):
    root = os.path.dirname(ann_file)
    dataset_name = os.path.basename(ann_file).replace(".pkl", "")
    ind = dataset_name.rfind("_")
    dataset_name = dataset_name[:ind]
    syn_id_paths = glob.glob(os.path.join(root, "*"))
    syn_id_paths = [syn_id_path for syn_id_path in syn_id_paths if "ShapeNet" not in os.path.basename(syn_id_path)]

    poses = np.load("./datasets/ShapeNet/renders/poses.npy")

    affinity = []
    for pose in poses:
        affinity.append(np.argsort(haversine(pose, poses)))
    affinity = np.array(affinity)
    ids, affinity = BASELINES[dataset_name](affinity)

    _, id_range = affinity.shape
    source_ids = np.tile(ids, [id_range, 1]).T
    ids = np.stack([source_ids.ravel(), affinity.ravel()], axis=-1)
    ids = np.core.defchararray.add(ids.astype(np.str), ".png")

    syn_id_paths = sorted(syn_id_paths)
    train_number = len(syn_id_paths) // 5 * 4
    train_syn_id_paths = syn_id_paths[:train_number]
    test_syn_id_paths = syn_id_paths[train_number:]
    train_set = np.concatenate([np.core.defchararray.add(syn_id_path+"/", ids) for syn_id_path in train_syn_id_paths])
    val_set = np.concatenate([np.core.defchararray.add(syn_id_path+"/", ids) for syn_id_path in test_syn_id_paths])

    save_pickle(train_set, os.path.join(root, dataset_name+"_train.pkl"))
    save_pickle(val_set, os.path.join(root, dataset_name+"_val.pkl"))

    data_set = train_set if "train" in ann_file else val_set
    return data_set


ShapeNetData = Registry()


@ShapeNetData.register("IMAGE")
def get_image(cfg, rgb1_path, rgb2_path):
    img1 = Image.open(rgb1_path)
    img2 = Image.open(rgb2_path)
    scale = get_scale(cfg)
    width, height = img1.size
    width = width // scale
    height = height // scale
    img1 = img1.resize((width, height))
    img2 = img2.resize((width, height))
    img1 = np.array(img1)[:, :, :3]
    img2 = np.array(img2)[:, :, :3]
    image = np.concatenate([img1, img2], axis=2)
    return image


@ShapeNetData.register("MASK")
def get_mask(cfg, rgb1_path, rgb2_path):
    dmap1_path = rgb1_path.replace(".png", "") + "_depth.png"
    dmap2_path = rgb2_path.replace(".png", "") + "_depth.png"

    dmap1 = ShapeNetDB.read_depth(dmap1_path)
    dmap2 = ShapeNetDB.read_depth(dmap2_path)
    mask1 = Image.fromarray((dmap1 != 10).astype(np.int32))
    mask2 = Image.fromarray((dmap2 != 10).astype(np.int32))

    scale = get_scale(cfg)
    width, height = mask1.size
    width = width // scale
    height = height // scale
    mask1 = np.array(mask1.resize((width, height)))
    mask2 = np.array(mask2.resize((width, height)))
    mask = np.stack([mask1, mask2], axis=2)
    return mask


class ShapeNetDataset(Dataset):
    intrinsic_matrix = {
        "blender": np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }

    def __init__(self, cfg, ann_file, transforms=None):
        self.cfg = cfg
        self.ids = ShapeNetDB.get_ids(ann_file)
        self.ann_file = ann_file

    def get_data(self, idx, data_name):
        rgb1_path, rgb2_path = self.ids[idx]
        data = ShapeNetData[data_name](self.cfg, rgb1_path, rgb2_path)
        return data

    def __getitem__(self, idx):
        rgb1_path, rgb2_path = self.ids[idx]

        path_pattern = rgb1_path.replace(".png", "")
        dmap1_path = path_pattern + "_depth.png"
        pose1_path = path_pattern + "_RT.pkl"

        path_pattern = rgb2_path.replace(".png", "")
        dmap2_path = path_pattern + "_depth.png"
        pose2_path = path_pattern + "_RT.pkl"

        img1, dmap1, pose1 = ShapeNetDB.read_data(rgb1_path, dmap1_path, pose1_path)
        img2, dmap2, pose2 = ShapeNetDB.read_data(rgb2_path, dmap2_path, pose2_path)

        # TODO: collecting targets online makes that the speed of reading data reduces from 30 it/s to 20 it/s
        target = TARGETS[self.cfg.MODEL.TARGET](self.cfg, img1, img2, dmap1, dmap2, pose1, pose2,
                                                idx=idx, ann_file=self.ann_file)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        image = torch.cat((img1, img2), dim=0)

        return image, target, idx

    def __len__(self):
        return len(self.ids)
