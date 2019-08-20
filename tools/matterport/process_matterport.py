import numpy as np
import os
import imageio
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


def read_config_file(filename):
    f = open(filename, 'r')
    intrinsics = []
    depth_name = []
    image_name = []
    extrinsics = []
    for line in f:
        line_split = line.split()
        if line_split:
            if line_split[0] == 'intrinsics_matrix':
                intrinsics.append(np.array(line_split[1:]).astype(np.float))
            if line_split[0] == 'scan':
                depth_name.append(line_split[1])
                image_name.append(line_split[2])
                extrinsics.append(np.array(line_split[3:]).astype(np.float))
    intrinsics = np.array(intrinsics)
    extrinsics = np.array(extrinsics)

    return depth_name, image_name, intrinsics, extrinsics


def read_overlap_file(filename):
    f = open(filename, 'r')
    pairs = []
    overlap_infos = []
    for line in f:
        line_split = line.split()
        if line_split:
            if line_split[0] == 'II':
                pair = np.array(line_split[1:3]).astype(int)
                pairs.append(pair)
                overlap_info = np.array(line_split[3:]).astype(np.float)
                overlap_infos.append(overlap_info)

    pairs = np.array(pairs)
    overlap_infos = np.array(overlap_infos)

    return pairs, overlap_infos


def get_correspondence(imgname1, depth1, depth2, normal1, normal2, extrinsics1, extrinsics2, intrinsics1, intrinsics2):
    '''
    generate correspondences by directly projecting pixels in image 1 to pixels in image 2
    '''
    nrow, ncol = depth1.shape

    # read extrinsics and intrinsics parameters
    extrinsics1 = np.reshape(extrinsics1, (4, 4))
    extrinsics2 = np.reshape(extrinsics2, (4, 4))
    intrinsics2 = np.reshape(intrinsics2, (3, 3))
    extrinsics2_inv = np.linalg.inv(extrinsics2)

    # backproject pixels in image 1 into 3D points w.r.t. camera 1
    coord_row, coord_col = np.mgrid[0:nrow, 0:ncol]
    fx, fy, cx, cy = intrinsics1[0], intrinsics1[4], intrinsics1[2], intrinsics1[5]
    coord3d1x = 1.0 * (coord_col - cx) / fx * depth1
    coord3d1y = 1.0 * (coord_row - cy) / fy * depth1
    coord3d_11 = np.stack((coord3d1x, coord3d1y, depth1, np.ones((nrow, ncol))), axis=-1)
    coord3d_11 = coord3d_11[:, :, :, np.newaxis]

    # transform into the coordinate system of camera 2 and project into image 2
    coord3d_12 = extrinsics2_inv.dot(extrinsics1).dot(coord3d_11)
    coord3d_12 = np.transpose(coord3d_12, (1, 2, 0, 3))
    depth12 = coord3d_12[:, :, 2, 0]
    coord12_hm = intrinsics2.dot(coord3d_12[:, :, :3, :])
    coord12z = coord12_hm[2, :, :, 0]
    coord12xy = coord12_hm[0:2, :, :, 0] / coord12z
    coord12x, coord12y = coord12xy[0, :, :], coord12xy[1, :, :]

    # depth evaluation
    threshold = 0.05  # threshold for depth evaluation
    depth12_gt = ndimage.map_coordinates(depth2, [coord12y, coord12x])  # get goundtruth depth in img2
    vis_depth = (depth1 > 0) & (depth12 <= (1 + threshold)*depth12_gt) & (depth12 >= (1 - threshold)*depth12_gt)

    # get and evaluate surface normal
    normal1 = cv2.resize(normal1, dsize=(1280, 1024))
    normal1_norm = np.sum(normal1**2, axis=-1)**0.5
    normal1_valid = normal1_norm < 1.05
    normal2 = cv2.resize(normal2, dsize=(1280, 1024))
    rotation1 = extrinsics1[:3, :3]
    rotation2 = extrinsics2[:3, :3]
    relative_rotation = rotation2.transpose().dot(rotation1)
    normal12x_gt = ndimage.map_coordinates(normal2[:, :, 0], [coord12y, coord12x])
    normal12y_gt = ndimage.map_coordinates(normal2[:, :, 1], [coord12y, coord12x])
    normal12z_gt = ndimage.map_coordinates(normal2[:, :, 2], [coord12y, coord12x])
    normal12_gt = np.array([normal12x_gt, -normal12y_gt, -normal12z_gt])
    normal1[:, :, 1:3] = -normal1[:, :, 1:3]
    normal12 = relative_rotation.dot(normal1[:, :, :, np.newaxis])[:, :, :, 0]
    normal_innerproduct = np.sum(normal12 * normal12_gt, axis=0)
    vis_normal = (normal_innerproduct > 0.3) & normal1_valid

    # prune out texture-less regions
    img1 = cv2.imread(imgname1, 0)
    bd = 5  # padding size
    # gray_blur = cv2.GaussianBlur(img1, (5, 5), 2)
    gray_blur = cv2.bilateralFilter(img1, 9, 75, 75)
    # lap = cv2.Laplacian(gray_blur[bd:-bd, bd:-bd], cv2.CV_16S, ksize=5)
    # sobel = cv2.Canny(gray_blur[bd:-bd, bd:-bd], 1, 30, apertureSize=3)
    har = cv2.cornerHarris(gray_blur[bd:-bd, bd:-bd], 3, 3, 0.04)
    har = (np.absolute(har) > np.max([0.001*har.max(), 5e-8]))
    # plt.imshow(lap), plt.show()
    binarymap = np.zeros((1024, 1280))
    binarymap[bd:-bd, bd:-bd] = np.absolute(har).astype(np.float32)
    # kernel_e = np.ones((3, 3), np.uint8)
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # binarymap_e = cv2.erode(binarymap, kernel_e, iterations=1)
    vis_texture = cv2.dilate(binarymap, kernel_d, iterations=1)
    # plt.imshow(binarymap_d), plt.show()

    # get final visualization mask
    vis_fov = (coord12x >= 0) & (coord12y >= 0) & (coord12x <= ncol - 1) & (coord12y <= nrow - 1)
    vis_all = vis_fov * vis_depth * vis_normal * vis_texture
    ratio = np.sum(vis_all) / (1280.*1024.)
    return np.flip(coord12xy, 0), vis_all, ratio


def write_correspondence(Path, house_name, imgname1, imgname2, correspondence, visibility):
    fname = os.path.join(Path, "%s-%s-%s.png" % (house_name, imgname1, imgname2))
    valid_corspd = (50 * correspondence * visibility).transpose(1, 2, 0).astype(np.uint16)
    third_channel = np.zeros((1024, 1280), dtype=np.uint16)
    png_corspd = np.dstack((valid_corspd, third_channel))
    imageio.imwrite(fname, png_corspd, format='PNG-FI')


def view_correspondence(house_name, ind1, ind2, imgname1, imgname2, corspd, vis1):
    '''
    view correspondences generated by directly projecting pixels in image 1 to pixels in image 2
    '''
    img1 = cv2.imread(imgname1)
    img2 = cv2.imread(imgname2)
    nrow, ncol, nchn = img1.shape
    coord1 = np.array(np.where(vis1))
    coord2 = corspd[:, coord1[0], coord1[1]]

    # overlap region in image 2
    vis2 = np.zeros((nrow, ncol), dtype=bool)
    vis2[coord2[0].astype(int), coord2[1].astype(int)] = True
    _, num_match = coord1.shape
    kpts1 = [cv2.KeyPoint(coord1[1, i], coord1[0, i], 1) for i in range(num_match)]
    kpts2 = [cv2.KeyPoint(coord2[1, i], coord2[0, i], 1) for i in range(num_match)]

    # create sparse keypoints to show
    down_sample_factor = 50
    vis1_ds = vis1[::down_sample_factor, ::down_sample_factor]
    corspd_ds = corspd[:, ::down_sample_factor, ::down_sample_factor]
    coord1_ds = np.array(np.where(vis1_ds))
    coord2_ds = corspd_ds[:, coord1_ds[0], coord1_ds[1]]
    _, num_show = coord1_ds.shape
    kpts1_show = [cv2.KeyPoint(down_sample_factor*coord1_ds[1, i], down_sample_factor*coord1_ds[0, i], 1)
                  for i in range(num_show)]
    kpts2_show = [cv2.KeyPoint(coord2_ds[1, i], coord2_ds[0, i], 1)
                  for i in range(num_show)]
    matches = [cv2.DMatch(i, i, 0) for i in range(num_show)]

    # create mask for showing overlapping regions
    mask1 = np.zeros((nrow, ncol, nchn))
    mask2 = np.zeros((nrow, ncol, nchn))
    mask1[:, :, 2] = 60 * np.ones((nrow, ncol)) * vis1
    mask2[:, :, 2] = 60 * np.ones((nrow, ncol)) * vis2
    img1_show = np.clip(mask1+img1, 0, 255).astype(np.uint8)
    img2_show = np.clip(mask2+img2, 0, 255).astype(np.uint8)
    outImg = np.array([])
    outImg = cv2.drawMatches(img1_show, kpts1_show, img2_show, kpts2_show, matches, outImg)

    # plot and save
    # plt.imshow(outImg); plt.show()
    savefolder = '/tmp/vis'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    cv2.imwrite('%s/%s_%d_%d.png'
                % (savefolder, house_name, ind1, ind2), outImg)


def compute_all(data_path, save_path, house_name):
    folder_path = os.path.join(data_path, house_name, house_name)
    img_path = os.path.join(folder_path, "undistorted_color_images/")
    depth_path = os.path.join(folder_path, "undistorted_depth_images/")
    normal_path = os.path.join(folder_path, "undistorted_normal_images/")
    config_file = os.path.join(folder_path, "undistorted_camera_parameters/{}.conf".format(house_name))
    overlap_file = os.path.join(folder_path, "image_overlap_data/{}_iip.txt".format(house_name))

    depth_name, image_name, intrinsics, extrinsics = read_config_file(config_file)
    pairs, _ = read_overlap_file(overlap_file)

    for pair in pairs:
        ind1, ind2 = pair[0], pair[1]
        if ind1 == ind2 or ind1/18 == ind2/18:
            continue
        img_name1, depth_name1, extrinsic1, intrinsic1 = \
            image_name[ind1], depth_name[ind1], extrinsics[ind1], intrinsics[int(ind1/6)]
        img_name2, depth_name2, extrinsic2, intrinsic2 = \
            image_name[ind2], depth_name[ind2], extrinsics[ind2], intrinsics[int(ind2/6)]

        # correct extrinsics
        mask = np.tile(np.array([1, -1, -1, 1]), 4)
        extrinsic1 = extrinsic1 * mask
        extrinsic2 = extrinsic2 * mask
        extrinsic1 = np.reshape(extrinsic1, (4, 4))
        extrinsic2 = np.reshape(extrinsic2, (4, 4))
        relative_pose = np.linalg.inv(extrinsic2).dot(extrinsic1)
        relative_dis = np.sqrt(np.sum(relative_pose[0:3, 3]**2))
        # get depth uint16 and convert into meters
        depth1_uint16 = imageio.imread(depth_path + depth_name1)
        depth2_uint16 = imageio.imread(depth_path + depth_name2)
        depth1 = np.array(depth1_uint16, dtype=np.float) / 4000
        depth2 = np.array(depth2_uint16, dtype=np.float) / 4000

        # get normal unit16 and convert
        normal1x_uint16 = imageio.imread(normal_path + depth_name1[:-4] + "_nx.png")
        normal1y_uint16 = imageio.imread(normal_path + depth_name1[:-4] + "_ny.png")
        normal1z_uint16 = imageio.imread(normal_path + depth_name1[:-4] + "_nz.png")
        normal1 = np.stack((normal1x_uint16, normal1y_uint16, normal1z_uint16), -1).astype(np.float)/32768 - 1
        normal2x_uint16 = imageio.imread(normal_path + depth_name2[:-4] + "_nx.png")
        normal2y_uint16 = imageio.imread(normal_path + depth_name2[:-4] + "_ny.png")
        normal2z_uint16 = imageio.imread(normal_path + depth_name2[:-4] + "_nz.png")
        normal2 = np.stack((normal2x_uint16, normal2y_uint16, normal2z_uint16), -1).astype(np.float) / 32768 - 1

        # get pixel correspondences img1->img2
        correspondence, visibility, ratio = get_correspondence(img_path + img_name1, depth1, depth2, normal1, normal2,
                                                               extrinsic1, extrinsic2, intrinsic1, intrinsic2)
        # valid only if pixels having correspondences exceed ratio
        if ratio > 0.001:
            print("generate correspondences between image %d and %d in house %s" % (ind1, ind2, house_name))
            view_correspondence(house_name, ind1, ind2, img_path + img_name1, img_path + img_name2,
                                correspondence, visibility)
            # write_correspondence(save_folder, house_name, img_name1, img_name2, correspondence, visibility)
            #
            # correspondence_inv, visibility_inv, _ = get_correspondence(imgpath + imgname2, depth2, depth1, normal2,
            #                                                            normal1, extrinsic2, extrinsic1, intrinsic2,
            #                                                            intrinsic1)
            # write_correspondence(savefolder, house_name, imgname2, imgname1, correspondence_inv, visibility_inv)
        print(ratio)

    import ipdb; ipdb.set_trace()


def read_match_file(img_path, filename):
    scale_x = 640 / 1280
    scale_y = 512 / 1024
    img_pattern = img_path + "color-{:02d}-{:06d}.jpg"
    with open(filename, "r") as f:
        for line1 in f:
            line1_split = line1.split()
            if len(line1_split) == 0:
                continue
            if not line1_split[0].isdigit():
                continue
            img_name1 = img_pattern.format(int(line1_split[1]), int(line1_split[2]))
            pixel1 = (int(line1_split[3]) * scale_x, int(line1_split[4]) * scale_y)

            line2 = f.readline()
            line2_split = line2.split()
            img_name2 = img_pattern.format(int(line2_split[1]), int(line2_split[2]))
            pixel2 = (int(line2_split[3]) * scale_x, int(line2_split[4]) * scale_y)

            img1 = cv2.imread(img_name1)
            img2 = cv2.imread(img_name2)
            kpts1 = [cv2.KeyPoint(pixel1[0], pixel1[1], 1)]
            kpts2 = [cv2.KeyPoint(pixel2[0], pixel2[1], 1)]
            match = [cv2.DMatch(0, 0, 0)]
            out_img = cv2.drawMatches(img1, kpts1, img2, kpts2, match, None)
            plt.imshow(out_img)
            plt.show()

        import ipdb; ipdb.set_trace()


def read_official_corspd():
    data_path = "./data/MATTERPORT/v1/tasks/keypoint_matching/data/17DRP5sb8fy/"
    match_file = data_path + "matches.txt"
    img_path = data_path + "images/"
    read_match_file(img_path, match_file)


if __name__ == "__main__":
    # read_official_corspd()
    house_names = ["gTV8FGcVJC9", "JF19kD82Mey", "X7HyMhZNoso"]
    data_path = "./data/MATTERPORT/v1/scans"
    save_path = "./data/MATTERPORT/processedDataset"
    compute_all(data_path, save_path, house_names[0])
