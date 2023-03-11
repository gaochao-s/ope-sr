import os
import time
import shutil
import math
import torch.nn as nn
import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


def log_fn(obj, log_path, filename='log.txt'):
    print(obj)
    with open(os.path.join(log_path, filename), 'a') as f:
        print(obj, file=f)


def ensure_path(path, pause_start):
    # basename = os.path.basename(path.rstrip('/'))
    if pause_start:
        os.makedirs(path, exist_ok=True)
    elif os.path.exists(path):
        if input('{} exists, remove? (y/[n]): '.format(path)) == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, pause_start=False, log_name='log.txt', writer=True):
    ensure_path(save_path, pause_start)
    if writer is True:
        writer = SummaryWriter(os.path.join(save_path, 'runs'))
    else:
        writer = None
    log = lambda obj, log_path=save_path, filename=log_name: log_fn(obj, log_path, filename)
    return log, writer


def set_log_path(save_path, log_name='log.txt'):
    log = lambda obj, log_path=save_path, filename=log_name: log_fn(obj, log_path, filename)
    return log


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_loss_fn(loss_fn_spec):
    loss_list = {
        'l1': nn.L1Loss(),
        'l2': nn.MSELoss(),
    }
    loss_fns = []
    for tmp_loss_name, weight in zip(loss_fn_spec['name'], loss_fn_spec['weight']):
        tmp_loss = loss_list[tmp_loss_name]
        loss_fns.append(lambda pred, gt: weight * tmp_loss(pred, gt))
    return loss_fns


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    return ckpt

if __name__ == '__main__':
    log_1, writer1 = set_save_path('./save_1')
    log_2, writer2 = set_save_path('./save_2')
    log_1('xdfa')
    log_2('dafe')
