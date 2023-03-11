import argparse
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import datasets
import models
from test import eval_psnr_ope
import dl_utils


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    data0 = dataset[0]
    info = []
    for k, v in data0.items():
        info.append(f'{k}:{v.shape}')
    print(', '.join(info))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=16, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('finetune') is not None:  # for finetune
        sv_file = torch.load(config['finetune'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = dl_utils.make_optimizer(
            model.parameters(), config['optimizer'])
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        epoch_start = 1
    elif pause_start:
        sv_file = torch.load(last_path)
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = dl_utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    elif config.get('resume') is not None:  # for resume from epoch xxx
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = dl_utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()

    else:  # for start
        model = models.make(config['model']).cuda()
        optimizer = dl_utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    # log('model: #params={}'.format(dl_utils.compute_num_params(model, text=True)))

    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    train_loss = dl_utils.Averager()
    loss_fn = nn.L1Loss()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        pred_sample = model(img_lr=batch['lr_img'], sample_coords=batch['coords_sample'])
        pred_sample.clamp_(-1, 1)
        loss = loss_fn(pred_sample, batch['gt_sample'])
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = dl_utils.set_save_path(save_path, pause_start, writer=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_psnr = -1e18
    max_val_ssim = -1e18

    timer = dl_utils.Timer()
    # gpu_tracker.track()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))

        writer.add_scalars('train_loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == -1):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            with torch.no_grad():
                val_psnr, val_ssim = eval_psnr_ope(val_loader, model_)

            log_info.append('val: psnr={:.4f}'.format(val_psnr))
            log_info.append('val: ssim={:.4f}'.format(val_ssim))
            writer.add_scalars('psnr', {'val': val_psnr}, epoch)
            writer.add_scalars('ssim', {'val': val_ssim}, epoch)
            if val_psnr > max_val_psnr:
                max_val_psnr = val_psnr
                torch.save(sv_file, os.path.join(save_path, f'epoch-best-psnr.pth'))
                log_info.append('get best psnr')
            if val_ssim > max_val_ssim:
                max_val_ssim = val_ssim
                torch.save(sv_file, os.path.join(save_path, f'epoch-best-ssim.pth'))
                log_info.append('get best ssim')

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = dl_utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = dl_utils.time_text(t), dl_utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train-div2k-configs/train_edsr-ope-1.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default='exp_01')
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--pause_start', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    global pause_start, last_path
    pause_start = args.pause_start
    last_path = os.path.join(save_path, 'epoch-last.pth')

    main(config, save_path)
