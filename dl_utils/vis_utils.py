from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import PIL.Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def bicubic_scale(img_path, scale, save_path):
    """
    .png to .png
    """
    img = load_imageToten(img_path)
    img_name = img_path.split('/')[-1].split('.')[0]
    h, w = img.shape[-2:]
    print(f'img: {img_name}, h: {h}, w: {w}')

    target_h = int(h // scale)
    target_w = int(w // scale)
    img_target = load_imageToten(img_path, resize=(target_h, target_w))
    print(f'img: {img_name}, target_h: {target_h}, target_w: {target_w}')

    save_name = f'{img_name}_{target_h}x{target_w}.png'
    os.makedirs(save_path, exist_ok=True)
    save_tenimage(img_target, save_path, save_name)


def cut_img(loadpath, h_1=0, h_2=1, w_1=0, w_2=1):
    """
    .png to .png
    """
    save_path = 'test_imgs/pre_cut'
    os.makedirs(save_path, exist_ok=True)
    img = load_imageToten(loadpath)
    img_name = loadpath.split('/')[-1].split('.')[0]
    h, w = img.shape[-2:]
    h_up = int(h * h_1)
    h_down = int(h * h_2)
    w_left = int(w * w_1)
    w_right = int(w * w_2)
    img = img[:, :, h_up:h_down, w_left:w_right]
    cut_h, cut_w = img.shape[-2:]
    save_name = f'{img_name}_cut_{cut_h}x{cut_w}.png'
    save_tenimage(img, save_path, save_name)


def cut_img_dir(img, h_1=0, h_2=1, w_1=0, w_2=1):
    """
    input: tensor [1,3,H,W];
    """
    h, w = img.shape[-2:]
    h_up = int(h * h_1)
    h_down = int(h * h_2)
    w_left = int(w * w_1)
    w_right = int(w * w_2)
    img = img[:, :, h_up:h_down, w_left:w_right]
    cut_h, cut_w = img.shape[-2:]
    return img, [cut_h, cut_w]


def near_scale(loadpath, h, w):
    """
    .png to tensor [1,3,H,W]
    """
    img = Image.open(loadpath)
    transform_near = transforms.Compose([
        transforms.Resize((h, w), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img_res = transform_near(img).unsqueeze(0)
    return img_res


def resize_img(img, size, norm=True):
    """
    input: tensor [1,3,H,W]; value in [-1,1]
    return: tensor [1,3,size,size]; value in [-1,1]
    """
    if norm:
        img = (img + 1) / 2
    ans = transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img.squeeze(0)))).unsqueeze(0)
    return (ans - 0.5) / 0.5

def resize_img_near(img, size, norm=True):
    """
    input: tensor [1,3,H,W]; value in [-1,1]
    return: tensor [1,3,size,size]; value in [-1,1]
    """
    if norm:
        img = (img + 1) / 2
    ans = transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.NEAREST)(
            transforms.ToPILImage()(img.squeeze(0)))).unsqueeze(0)
    return (ans - 0.5) / 0.5

def save_tenimage(imgTensor, svpath, svname, norm=True):
    """
    input: [1,3,H,W]; value in [-1,1]
    """
    utils.save_image(
        imgTensor,
        os.path.join(svpath, svname),
        nrow=1,
        normalize=norm,
        value_range=(-1, 1),
    )

def load_imageToten(loadpath, resize=None):
    """
    from load path to load image to tensor
    return: [1,3,H,W]; value in [-1,1]
    """
    img = Image.open(loadpath).convert('RGB')

    if resize is not None:
        if isinstance(resize, tuple):
            transform_bicub = transforms.Compose([
                transforms.Resize(resize, interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        if isinstance(resize, int):
            transform_bicub = transforms.Compose([
                transforms.Resize((resize, resize), interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    else:
        transform_bicub = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    img_res = transform_bicub(img).unsqueeze(0)
    return img_res

def loss_curve(loss_list):
    epochs_list = np.arange(len(loss_list)) + 1

    plt.plot(epochs_list, loss_list, label="loss")
    plt.xlabel('freq')
    plt.ylabel('value')
    plt.legend(loc=0, ncol=1)  # 参数：loc设置显示的位置，0是自适应；ncol设置显示的列数

    plt.show()
