import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from dl_utils import to_pixel_samples
from torchvision.transforms import InterpolationMode


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


# for liif/lte train/eval
@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)]  # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


####### train
# ope sample for train
@register('ope-sample-train')  # abandoned
class OPE_sample(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=True, norm=True, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.norm = norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        ### prepare sample
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = resize_fn(crop_hr, w_lr)  # img_lr_s

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            hr_rgb = (hr_rgb - 0.5) / 0.5

        sample_batch = {
            'lr_img': crop_lr,
            'coords_sample': hr_coord,
            'gt_sample': hr_rgb
        }

        return sample_batch

@register('ope-sample-train-2')
class OPE_sample_2(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=True, norm=True, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.norm = norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        ### prepare sample
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = resize_fn(crop_hr, w_lr)  # img_lr_s

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        crop_hr = resize_fn(crop_hr, w_hr * 3)
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            hr_rgb = (hr_rgb - 0.5) / 0.5

        sample_batch = {
            'lr_img': crop_lr,
            'coords_sample': hr_coord,
            'gt_sample': hr_rgb
        }

        return sample_batch


####### eval
@register('ope-patch-eval')  # use for eval patch
class OPE_patch(Dataset):
    def __init__(self, dataset, inp_size=None, scale_factor=None, augment=False, norm=True):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.norm = norm
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        # crop_size = int(self.inp_size * (1 + random.random()))
        crop_size = int(self.inp_size * self.scale_factor)
        img_h, img_w = img.shape[-2:]
        if crop_size >= min(img_h, img_w):
            crop_size_t = min(img_h, img_w)
            x0 = random.randint(0, img_h - crop_size_t)
            y0 = random.randint(0, img_w - crop_size_t)
            img_hr = img[:, x0:x0 + crop_size_t, y0:y0 + crop_size_t]
            img_hr = resize_fn(img_hr, crop_size)
            img_lr = resize_fn(img_hr, self.inp_size)
        else:
            x0 = random.randint(0, img_h - crop_size)
            y0 = random.randint(0, img_w - crop_size)
            img_hr = img[:, x0:x0 + crop_size, y0:y0 + crop_size]
            img_lr = resize_fn(img_hr, self.inp_size)
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img_lr = augment(img_lr)
            img_hr = augment(img_hr)

        if self.norm:
            img_lr = (img_lr - 0.5) / 0.5
            img_hr = (img_hr - 0.5) / 0.5

        return {
            'lr': img_lr,  # C,h,w
            'hr': img_hr,  # C,H,W
        }


####### test
@register('sr-cut-paired')  # use for test div2k/benchmark x2/x3/x4
class SRcutPaired(Dataset):

    def __init__(self, dataset, inp_size=None, norm=True, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.norm = norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            crop_hr = (crop_hr - 0.5) / 0.5

        return {
            'lr': crop_lr,
            'gt': crop_hr,
        }


@register('sr-cut-downsampled-test')  # use for div2k/benchmark x6 x8 ... x20
class CutDownsampledTest(Dataset):
    def __init__(self, dataset, test_scale=None, augment=False, norm=True):
        self.dataset = dataset
        self.augment = augment
        self.norm = norm
        self.test_scale = test_scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = self.test_scale
        h_lr = math.floor(img.shape[-2] / s + 1e-9)
        w_lr = math.floor(img.shape[-1] / s + 1e-9)
        img = img[:, :round(h_lr * s), :round(w_lr * s)]
        img_down = resize_fn(img, (h_lr, w_lr))
        crop_lr, crop_gt = img_down, img

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_gt = augment(crop_gt)

        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            crop_gt = (crop_gt - 0.5) / 0.5

        return {
            'lr': crop_lr,
            'gt': crop_gt,
        }
