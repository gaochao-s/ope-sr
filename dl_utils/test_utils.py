import json
import torch
from .train_utils import make_coord


def load_json(path):
    with open(path, encoding='utf8') as f:
        target_dic = json.load(f)
    return target_dic

def save_json(path, save_dic):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(save_dic, f, ensure_ascii=False, indent=2)

def batched_predict(model, inp, coord, cell, bsize):
    model.gen_feat(inp)
    n = coord.shape[1]
    ql = 0
    preds = []
    while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
        preds.append(pred)
        ql = qr
    pred = torch.cat(preds, dim=1)
    return pred


def pixel_gen_img(batch_lr, model, h, w, bsize=30000):
    coord = make_coord((h, w)).cuda()
    N = batch_lr.shape[0]
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, batch_lr,
                           coord.unsqueeze(0).repeat(N, 1, 1), cell.unsqueeze(0).repeat(N, 1, 1), bsize)
    pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)
    return pred