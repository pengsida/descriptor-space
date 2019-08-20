"""
Any other useful tools.
"""
import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from skimage import io

def toheapmap_torch(gray):
    """
    Turn a torch gray scale image to heatmap
    :param gray: (H, W), value in (0, 1)
    :return: torch image (3, H, W)
    """
    import matplotlib as mpl
    mpl.use('Agg')
    gray = tonumpy(gray)
    plt.imshow(gray, cmap=plt.cm.hot_r)
    plt.colorbar()
    plt.clim(0.2, 0.4)
    plt.savefig('temp.png')
    plt.clf()
    image = io.imread('temp.png')[:, :, :3]

    image = tofloat(image)
    
    return totensor(image)

def color_scale(attention):
    """
    Visualize a attention map
    :param scale_map: (C, H, W), attention map, softmaxed
    :return: (3, H, W), colored version
    """
    
    colors = torch.Tensor([
        [1, 0, 0], # red
        [0, 1, 0], # green
        [0, 0, 1], # blue
        [0, 0, 0], # black
    ]).float()
    
    # (H, W)
    attention = torch.argmax(attention, dim=0)
    # (H, W, C)
    color_map = colors[attention]
    color_map = color_map.permute(2, 0, 1)
    
    return color_map
    

def warp_torch(map, H):
    """
    Warp a torch image.
    :param map: either (C, H, W) or (H, W)
    :param H: (3, 3)
    :return: warped iamge, (C, H, W) or (H, W)
    """
    map = tonumpy(map)
    h, w = map.shape[-2:]
    map = cv2.warpPerspective(map, H, dsize=(w, h))
    
    return totensor(map)

def torange(array, low, high):
    """
    Render an array to value range (low, high)
    :param array: any array
    :param low, high: the range
    :return: new array
    """
    min, max = array.min(), array.max()
    # normalized to [0, 1]
    array = array - min
    array = array / (max - min)
    # to (low, high)
    array = array * (high - low) + low
    
    return array

def touint8(img):
    """
    Convert float numpy image to uint8 image
    :param img: numpy image, float, (0, 1)
    :return: uint8 image
    """
    img = img * 255
    return img.astype(np.uint8)

def tofloat(img):
    """
    Convert a uint8 image to float image
    :param img: numpy image, uint8
    :return: float image
    """
    return img.astype(np.float) / 255

def tonumpy(img):
    """
    Convert torch image map to numpy image map
    
    :param img: tensor, shape (C, H, W)
    :return: numpy, shape (H, W, C)
    """
    if len(img.size()) == 2:
        return img.cpu().detach().numpy()
    
    return img.permute(1, 2, 0).cpu().detach().numpy()

def tonumpy_batch(imgs):
    """
    Convert a batch of torch images to numpy image map
    
    :param imgs: (B, C, H, W)
    :return: (B, H, W, C)
    """
    
    return imgs.permute(0, 2, 3, 1).cpu().detach().numpy()

def totensor(img, device=torch.device('cpu')):
    """
    Do the reverse of tonumpy
    """
    if len(img.shape) == 2:
        return torch.from_numpy(img).to(device).float()
    return torch.from_numpy(img).permute(2, 0, 1).to(device).float()

def totensor_batch(imgs, device=torch.device('cpu')):
    """
    Do the reverse of tonumpy_batch
    """
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device).float()

def RGB2BGR(*imgs):
    return [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in imgs]

def unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized tensor image to unnormalized form
    :param img: (C, H, W)
    """
    img = img.detach().cpu().clone()
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    
    return img

def tensor2RGB(img):
    return (tonumpy(unnormalize(img)) * 255.).astype(np.uint8)

