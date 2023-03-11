# OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution

This is the official implementation for the paper OPE-SR: 
"OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution" (CVPR 2023)

## Environment
- python 3
- pytorch 1.8.1+cu111
- tensorboardX
- timm, imageio, yaml, tqdm, matplotlib

## Datasets
- download the datasets. see [LIIF](https://github.com/yinboc/liif).
- put the datasets as follows:
```
datasets
├── benchmark
│   ├── B100
│   │   ├── HR
│   │   ├── LR_bicubic
│   ├── Set14
│   │   ├── HR
│   │   ├── LR_bicubic
│   ├── Set5
│   │   ├── HR
│   │   ├── LR_bicubic
│   └── Urban100
│       ├── HR
│       ├── LR_bicubic
├── div2k
│   ├── DIV2K_train_HR
│   ├── DIV2K_valid_HR
│   ├── DIV2K_valid_LR_bicubic_X2
│   ├── DIV2K_valid_LR_bicubic_X3
│   └── DIV2K_valid_LR_bicubic_X4

```

## Checkpoints
- download the pre-trained encoder:

Model | Download
:-: | :-:
EDSR-OPE | [Google Drive](https://drive.google.com/drive/folders/1MyQTwobDiHd1v_eYSOKOIfFcprjUzBfi?usp=share_link)
RDN-OPE | [Google Drive](https://drive.google.com/drive/folders/13UpkVbj0IQDAqheUpzqPt_Ue9F-6PBkB?usp=share_link)

- mkdir save, put the checkpoint as follows:

```
save
├── train_edsr-ope
│   ├── epoch-1000.pth
├── train_rdn-ope
│   ├── epoch-1000.pth
```

## For Quick Demo
e.g. use EDSR encoder to test 0803.png in test_imgs folder for scale x4,x6,x8,x12, results will be saved in exp_folder.
```commandline
python  ope_demo_zoom.py --exp_folder save/train_edsr-ope --ckpt_name epoch-1000.pth --hr_path test_imgs/0803.png --scale_list 4 6 8 12
```

## For Reproducing Experiments
e.g. use EDSR encoder for div2k x12 scale.
```commandline
python test_auto.py --exp_folder save/train_edsr-ope --test_config configs/test-configs/test_CIR-SR-div2k-x12.yaml
```

## For Training
e.g. train EDSR encoder use GPU 0:
```commandline
python train_ope_arbSR.py --config configs/train-div2k-configs/train_edsr-ope-1.yaml --tag exp_01 --gpu 0
```
e.g. train RDN encoder use GPU 0,1:
```commandline
python train_ope_arbSR.py --config configs/train-div2k-configs/train_rdn-ope-1.yaml --tag exp_01 --gpu 0,1
```


## Acknowledgements
We would like to express our gratitude to [LIIF](https://github.com/yinboc/liif) and [LTE](https://github.com/jaewon-lee-b/lte) for their invaluable contribution to the development community. Our project is built upon them, which provided the foundation for much of our work. As beginners in this field, we learned a great deal from their project and were able to build upon it to create something new.

