import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random


class BaseDataset(data.Dataset):
    """
    Base class for a dataset used in machine learning tasks.
    Inherits from `torch.utils.data.Dataset`, which requires overriding
    `__getitem__` and `__len__` methods.
    """

    def __init__(self):
        """
        Constructor for BaseDataset class.
        Initializes the base class and prepares any necessary components.
        """
        super(BaseDataset, self).__init__()

    def name(self):
        """
        Returns the name of the dataset.
        This function is overridden in derived classes to provide a custom name.

        Returns:
            str: 'BaseDataset' as the name of the dataset.
        """
        return 'BaseDataset'

    def initialize(self, opt):
        """
        Placeholder method for initialization with options.

        Args:
            opt: Options or configuration parameters for dataset setup.
        """
        pass


def get_params(opt, size):
    """
    Generates transformation parameters for resizing and cropping based on the input options.

    Args:
        opt: Configuration options (e.g., resizing and cropping strategies).
        size: Tuple containing the current image size (width, height).

    Returns:
        dict: Dictionary containing crop position and flip state.
    """
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    # flip is set to 0, could be enabled based on random decision
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    """
    Returns a sequence of transformations to be applied to the input image.

    Args:
        opt: Configuration options.
        params: Transformation parameters such as cropping position and flip state.
        method: The resampling method used for resizing.
        normalize: Whether to normalize the image.

    Returns:
        torchvision.transforms.Compose: A composed transformation function.
    """
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        osize = [256, 192]
        transform_list.append(transforms.Resize(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    """
    Returns a normalization function for input images.

    Returns:
        torchvision.transforms.Normalize: A normalization function.
    """
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    """
    Resizes the image to the nearest power of 2 dimensions, based on the given base size.

    Args:
        img: The input image to be resized.
        base: The base size for resizing (nearest power of 2).
        method: The resampling method used for resizing.

    Returns:
        PIL.Image: The resized image.
    """
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    """
    Scales the image to a target width, adjusting the height to maintain the aspect ratio.

    Args:
        img: The input image to be resized.
        target_width: The target width for the resized image.
        method: The resampling method used for resizing.

    Returns:
        PIL.Image: The resized image.
    """
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    """
    Crops the image to the given size at the specified position.

    Args:
        img: The image to be cropped.
        pos: A tuple (x, y) representing the top-left corner of the crop.
        size: The size of the crop (width, height).

    Returns:
        PIL.Image: The cropped image.
    """
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    """
    Flips the image horizontally if flip is set to True.

    Args:
        img: The image to be flipped.
        flip: Boolean flag indicating whether to flip the image.

    Returns:
        PIL.Image: The flipped image if flip is True, otherwise the original image.
    """
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
