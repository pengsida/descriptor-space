"""
File pix2idx

This file contains utility for converting image data (i.e., of shape (H, W, C))
to flattened version (i.e., shape (H*W, C)), in a consistent way
"""

def flatten_image(img):
    """
    Flatten an image
    
    :param img: numpy, shape (H, W, C)
    :return: (H * W, C)
    """
    H, W, C = img.shape
    return img.reshape(H * W, C)

def unflatten_image(img, size):
    """
    Unflatten an image
    
    :param img: numpy, shape (H*W, C)
    :param size: (w, h)
    :return: shape (H, W, C)
    """
    w, h = size
    assert w * h == img.shape[0], 'In unflatten_image: shape mismatch'
    
    return img.reshape(h, w, -1)


    
