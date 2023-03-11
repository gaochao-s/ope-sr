import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
import datasets
import models
import dl_utils
from tensorboardX import SummaryWriter
from test import test_both_ope


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/train_rdn-ope')
    parser.add_argument('--test_config', default='configs/test-configs/test_CIR-SR-set14-x6.yaml')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_name = args.test_config.split('/')[-1].split('.')[-2]
    save_dir = os.path.join(args.exp_folder, 'TEST_folder/' + test_name)
    log, _ = dl_utils.set_save_path(save_dir, writer=False)
    os.makedirs(save_dir, exist_ok=True)


    ckpt_list = [int(ckpt_name.split('.')[0].split('-')[-1]) for ckpt_name in os.listdir(args.exp_folder) if
                 ckpt_name.endswith('0.pth') or ckpt_name.endswith('5.pth')]
    ckpt_list = sorted(ckpt_list)

    test_dic = {}
    if 'test_info.json' in os.listdir(save_dir):
        test_dic = dl_utils.load_json(path=os.path.join(save_dir, 'test_info.json'))
        tested_ckpt_list = [int(key) for key in test_dic.keys()]
        tested_ckpt_list = sorted(tested_ckpt_list)
    else:
        tested_ckpt_list = []

    for i in range(1, 1001):
        if i in ckpt_list and i not in tested_ckpt_list:
            # perform test
            ckpt_num = i
            print(f'testing ckpt: {ckpt_num}')
            ckpt_name = f'epoch-{ckpt_num}.pth'
            resume_path = os.path.join(args.exp_folder, ckpt_name)

            log_name = ckpt_name.split('.')[-2] + '_log.txt'

            with open(args.test_config, 'r') as f:
                test_config = yaml.load(f, Loader=yaml.FullLoader)

            test_spec = test_config['test_dataset']
            dataset = datasets.make(test_spec['dataset'])
            dataset = datasets.make(test_spec['wrapper'], args={'dataset': dataset})
            loader = DataLoader(dataset, batch_size=test_spec['batch_size'],
                                num_workers=8, pin_memory=True)

            sv_file = torch.load(resume_path, map_location=lambda storage, loc: storage)
            model = models.make(sv_file['model'], load_sd=True).cuda()

            test_psnr, test_ssim, test_run_time = test_both_ope(loader, model, log, log_name,
                                                                eval_type=test_config.get('eval_type'), up_down=test_config.get('up_down'))

            log('test avg: psnr={:.4f}'.format(test_psnr), filename=log_name)
            log('test avg: ssim={:.4f}'.format(test_ssim), filename=log_name)
            log(f'test avg encoder time: {test_run_time[0]}s', filename=log_name)
            log(f'test avg render time: {test_run_time[1]}s', filename=log_name)
            log(f'test avg all time: {test_run_time[2]}s', filename=log_name)

            test_dic.update({str(i): [test_psnr, test_ssim]})
            dl_utils.save_json(path=os.path.join(save_dir, 'test_info.json'), save_dic=test_dic)

    writer = SummaryWriter(os.path.join(save_dir, 'runs'))
    all_keys = sorted([int(key) for key in test_dic.keys()])
    for key in all_keys:
        writer.add_scalar('scalar/test_psnr', test_dic[str(key)][0], key)
        writer.add_scalar('scalar/test_ssim', test_dic[str(key)][1], key)
