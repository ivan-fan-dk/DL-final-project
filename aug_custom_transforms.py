from __future__ import division
import torch
import random
import numpy as np
from skimage.transform import resize

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0,2] = w - output_intrinsics[0,2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [resize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, output_intrinsics

class RandomGamma(object):
    """Apply random gamma shift."""
    def __init__(self, min_gamma=0.8, max_gamma=1.2):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        
    def __call__(self, sample):
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        for k in sample.keys():
            # Only apply to RGB images, not depth/pose
            if k.startswith('color'):
                sample[k] = sample[k] ** gamma
        return sample

class RandomBrightness(object):
    """Apply random brightness shift."""
    def __init__(self, min_bright=0.5, max_bright=2.0):
        self.min_bright = min_bright
        self.max_bright = max_bright
        
    def __call__(self, sample):
        bright = np.random.uniform(self.min_bright, self.max_bright)
        for k in sample.keys():
            if k.startswith('color'):
                sample[k] = sample[k] * bright
        return sample

class RandomColor(object):
    """Apply random color shift to each channel."""
    def __init__(self, min_factor=0.8, max_factor=1.2):
        self.min_factor = min_factor
        self.max_factor = max_factor
        
    def __call__(self, sample):
        for k in sample.keys():
            if k.startswith('color'):
                factors = np.random.uniform(self.min_factor, self.max_factor, 3)
                for i in range(3):
                    sample[k][:,:,i] = sample[k][:,:,i] * factors[i]
        return sample
