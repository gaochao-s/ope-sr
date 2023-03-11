import argparse
import os
import yaml
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import models
import dl_utils
from test import single_img_sr
from dl_utils import cut_img_dir, resize_img, resize_img_near


def main(ckpt_path, save_path, args):
    global log, writer
    log, _ = dl_utils.set_save_path(save_path, writer=False)

    sv_file = torch.load(ckpt_path)
    model = models.make(sv_file['model'], load_sd=True).cuda()

    img_hr = dl_utils.load_imageToten(args.hr_path).cuda()

    scale_list = args.scale_list
    cut_ratio = args.cut_ratio
    h, w = img_hr.shape[-2:]
    p_bar = tqdm(scale_list, leave=False)
    for scale in p_bar:
        p_bar.set_description(f'processing x{scale} ...')
        tmp_h = int(h // scale)
        tmp_w = int(w // scale)
        img_lr = dl_utils.load_imageToten(loadpath=args.hr_path, resize=(tmp_h, tmp_w)).cuda()
        sr_img, psnr_model, ssim_model, run_time = single_img_sr(img_lr, model, h=h, w=w, gt=img_hr)
        bic_sr_img = resize_img(img_lr, size=(h, w)).cuda()
        near_sr_img = resize_img_near(img_lr, size=(h, w)).cuda()
        metric_fn_psnr = dl_utils.calc_psnr
        metric_fn_ssim = dl_utils.calc_ssim
        psnr_bic = metric_fn_psnr(bic_sr_img, img_hr)
        ssim_bic = metric_fn_ssim(bic_sr_img, img_hr)
        log(f'scale: {scale}, psnr: {psnr_model}/{psnr_bic}, ssim: {ssim_model}/{ssim_bic}')

        sr_img_cut, cut_size1 = cut_img_dir(sr_img, cut_ratio[0], cut_ratio[1], cut_ratio[2], cut_ratio[3])
        bic_sr_img_cut, cut_size2 = cut_img_dir(bic_sr_img, cut_ratio[0], cut_ratio[1], cut_ratio[2], cut_ratio[3])
        near_sr_img_cut, cut_size3 = cut_img_dir(near_sr_img, cut_ratio[0], cut_ratio[1], cut_ratio[2], cut_ratio[3])
        dl_utils.save_tenimage(imgTensor=sr_img_cut, svpath=save_path,
                               svname=f'x{scale}_{cut_size1[0]}x{cut_size1[1]}_sr.png')
        dl_utils.save_tenimage(imgTensor=bic_sr_img_cut, svpath=save_path,
                               svname=f'x{scale}_{cut_size2[0]}x{cut_size2[1]}_bic.png')
        dl_utils.save_tenimage(imgTensor=near_sr_img_cut, svpath=save_path,
                               svname=f'x{scale}_{cut_size3[0]}x{cut_size3[1]}_near.png')
        dl_utils.save_tenimage(imgTensor=img_lr, svpath=save_path,
                               svname=f'x{scale}_{tmp_h}x{tmp_w}_input.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/train_edsr-ope')
    parser.add_argument('--ckpt_name', default='epoch-1000.pth')
    parser.add_argument('--hr_path', default='test_imgs/0803.png')
    parser.add_argument('--cut_ratio', default=[0, 1, 0, 1])
    parser.add_argument('--scale_list', nargs='+', type=int, default=[4, 6, 8, 12])
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ckpt_path = os.path.join(args.exp_folder, args.ckpt_name)

    ckpt_name_ = ckpt_path.split('/')[-1].split('.')[-2]
    img_hr_name = args.hr_path.split('/')[-1].split('.')[-2]
    sub_save_folder = 'SISR-zoom_folder/' + ckpt_name_ + '/' + img_hr_name
    save_path = os.path.join(args.exp_folder, sub_save_folder)

    main(ckpt_path, save_path, args)
