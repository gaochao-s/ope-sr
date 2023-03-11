import torch


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1, use_norm=True):
    """
    input: [-1,1] not normalize
    """
    if use_norm is True:
        sr = (sr + 1) / 2
        hr = (hr + 1) / 2
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
