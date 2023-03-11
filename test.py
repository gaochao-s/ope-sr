from functools import partial
import torch
from tqdm import tqdm
import dl_utils
import torch.nn as nn
import math


def check_updown(up_down):
    ud = up_down.split('-')[0]  # bic / avg / none
    ud_scale = int(up_down.split('-')[1])
    return ud, ud_scale


def eval_psnr_ope(loader, model):
    model.eval()

    metric_fn_psnr = dl_utils.calc_psnr

    val_res_psnr = dl_utils.Averager()

    pbar = tqdm(loader, leave=False, desc='eval_psnr')
    with torch.no_grad():
        for batch in pbar:
            batch_lr = batch['lr'].cuda()
            gt = batch['hr'].cuda()
            gt_size = gt.shape[2]
            pred, _ = model.inference(batch_lr, h=gt_size, w=gt_size)
            pred.clamp_(-1, 1)

            res_psnr = metric_fn_psnr(pred, gt)

            val_res_psnr.add(res_psnr.item(), gt.shape[0])

            pbar.set_description('val {:.4f}'.format(val_res_psnr.item()))

    return val_res_psnr.item(), 0


def eval_psnr(loader, model, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
              verbose=False):
    model.eval()

    if eval_type is None:
        metric_fn = dl_utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(dl_utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(dl_utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = dl_utils.Averager()

    pbar = tqdm(loader, leave=False, desc='eval_psnr')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]

            coord = dl_utils.make_coord((scale * (h_old + h_pad), scale * (w_old + w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0

            coord = batch['coord']
            cell = batch['cell']

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            if fast:
                pred = model(inp, coord, cell * max(scale / scale_max, 1))
            else:
                pred = dl_utils.batched_predict(model, inp, coord, cell * max(scale / scale_max, 1),
                                                eval_bsize)  # cell clip for extrapolation
        if type(pred) is tuple:
            pred = pred[0].clamp_(-1, 1)
        else:
            pred = pred.clamp_(-1, 1)

        if eval_type is not None and fast == False:  # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


def test_both_ope(loader, model, log_fn, log_name, eval_type=None, up_down=None):
    model.eval()
    ud = 'none'
    ud_scale = 1
    if up_down is not None:
        ud, ud_scale = check_updown(up_down)
    metric_fn_ssim = dl_utils.calc_ssim
    if eval_type is None:
        metric_fn_psnr = dl_utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn_psnr = partial(dl_utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn_psnr = partial(dl_utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res_psnr = dl_utils.Averager()
    val_res_ssim = dl_utils.Averager()
    avg_time_encoder = dl_utils.Averager()
    avg_time_render = dl_utils.Averager()
    avg_time_all = dl_utils.Averager()
    pbar = tqdm(loader, leave=False, desc='test_both')
    id = 0
    with torch.no_grad():
        for batch in pbar:
            torch.cuda.empty_cache()
            batch_lr = batch['lr'].cuda()
            gt = batch['gt'].cuda()
            gt_size = gt.shape[-2:]
            if ud == 'none':
                pred, run_time = model.inference(batch_lr, h=gt_size[0], w=gt_size[1])
                pred.clamp_(-1, 1)
            elif ud == 'bic':
                pred, run_time = model.inference(batch_lr, h=gt_size[0] * ud_scale, w=gt_size[1] * ud_scale)

                pred = dl_utils.resize_img(pred, (gt_size[0], gt_size[1])).cuda()
                pred.clamp_(-1, 1)
            elif ud == 'avg':
                pred, run_time = model.inference(batch_lr, h=gt_size[0] * ud_scale, w=gt_size[1] * ud_scale)
                m = nn.AdaptiveAvgPool2d((gt_size[0], gt_size[1]))
                pred = m(pred)
                pred.clamp_(-1, 1)

            else:
                RuntimeError('updown fault')

            res_psnr = metric_fn_psnr(pred, gt)
            res_ssim = metric_fn_ssim(pred, gt)
            log_fn(
                f'test_img: {id}, psnr: {res_psnr.item()}, ssim: {res_ssim.item()}, time: {run_time[0]}s/{run_time[1]}s/{run_time[2]}s',
                filename=log_name)
            val_res_psnr.add(res_psnr.item(), gt.shape[0])
            val_res_ssim.add(res_ssim.item(), gt.shape[0])
            avg_time_encoder.add(run_time[0], gt.shape[0])
            avg_time_render.add(run_time[1], gt.shape[0])
            avg_time_all.add(run_time[2], gt.shape[0])

            id += 1

            pbar.set_description('img:{}, psnr: {:.4f}, ssim: {:.4f}'.format(id - 1, res_psnr.item(), res_ssim.item()))

    return val_res_psnr.item(), val_res_ssim.item(), [avg_time_encoder.item(), avg_time_render.item(),
                                                      avg_time_all.item()]


def single_img_sr(lr_img, model, h, w, gt=None, up_down=None, flip=None):
    model.eval()
    ud = 'none'
    ud_scale = 1
    if up_down is not None:
        ud, ud_scale = check_updown(up_down)
    with torch.no_grad():
        if flip is not None:
            pred, run_time = model.inference(lr_img, h=h, w=w, flip_conf=flip)
            pred.clamp_(-1, 1)
        else:
            if ud == 'none':
                pred, run_time = model.inference(lr_img, h=h, w=w)
                pred.clamp_(-1, 1)
            elif ud == 'bic':
                pred, run_time = model.inference(lr_img, h=h * ud_scale, w=w * ud_scale)

                pred = dl_utils.resize_img(pred, (h, w)).cuda()
                pred.clamp_(-1, 1)
            elif ud == 'avg':
                pred, run_time = model.inference(lr_img, h=h * ud_scale, w=w * ud_scale)
                m = nn.AdaptiveAvgPool2d((h, w))
                pred = m(pred)
                pred.clamp_(-1, 1)

            else:
                RuntimeError('updown fault')

        if gt is not None:
            metric_fn_psnr = dl_utils.calc_psnr
            metric_fn_ssim = dl_utils.calc_ssim
            res_psnr = metric_fn_psnr(pred, gt)
            res_ssim = metric_fn_ssim(pred, gt)
            return pred, res_psnr, res_ssim, run_time
        else:
            return pred, None, None, run_time

