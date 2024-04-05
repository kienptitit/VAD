import numpy as np
import torch
from tqdm import tqdm
from model import *
from nf_utils import *
import random
from config import CFG
from dataset import get_dataloader
from torch.optim import lr_scheduler
import os
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from train_Stage_1 import nf_test_epoch
import pandas as pd
from torch.utils.data import DataLoader


def get_normal_samples(test_loader, gt):
    gt_tmp = torch.tensor(gt.copy()).cuda()
    normal_samples = []
    for i, (v_input) in tqdm(enumerate(test_loader)):
        n_snippet = v_input.shape[-2]

        v_input = v_input.float().cuda(non_blocking=True)[:, 1].squeeze()

        v_input = v_input.unsqueeze(1).repeat(1, 16, 1).reshape(-1, 1024)
        labels = gt_tmp[: n_snippet * 16]
        normal_samples.append(v_input[labels == 0].unique(dim=0))

        gt_tmp = gt_tmp[n_snippet * 16:]
    normal = torch.concat(normal_samples, dim=0)
    return normal


def get_normal_samples_2(test_loader, gt):
    gt_tmp = torch.tensor(gt.copy()).cuda()
    normal_samples = []
    abnormal_samples = []
    for i, (v_input) in tqdm(enumerate(test_loader)):
        n_snippet = v_input.shape[-2]

        v_input = v_input.float().cuda(non_blocking=True).mean(dim=1).squeeze()

        v_input = v_input.unsqueeze(1).repeat(1, 16, 1).reshape(-1, 1024)
        labels = gt_tmp[: n_snippet * 16]
        normal_samples.append(v_input[labels == 0].unique(dim=0))
        abnormal_samples.append(v_input[labels == 1].unique(dim=0))
        gt_tmp = gt_tmp[n_snippet * 16:]

    return normal_samples, abnormal_samples


if __name__ == '__main__':
    torch.manual_seed(22)
    # model_path = '/media/kiennguyen/Data/2023/NaverProject/UCF-Crime-10-Crop/nf.pt'
    model_path = '/home/kiennguyen/Downloads/NF_version_10C_mean.pt'
    args = CFG()
    # model_path = '/media/kiennguyen/New Volume/PTIT/Naver/VAD/NormalizingFlow/nf.pt'
    normalizing_flow = get_flow_model(args, args.in_features).to(args.device)
    normalizing_flow.load_state_dict(torch.load(model_path))
    pos_embed = PositionalEncoding1D(args.pos_embed_dim)  # Correct
    multi_gaussian = MultivariateNormal(torch.zeros(1024), torch.eye(1024))
    normalizing_flow.eval()

    _, test_loader = get_dataloader(args, mode='MGFN', backbone='rtfm')
    gt = np.load(args.ucf_gt)
    normal_samples, abnormal_samples = get_normal_samples_2(test_loader, gt)
    normal = torch.concat(normal_samples, dim=0)
    abnormal = torch.concat(abnormal_samples, dim=0)



    print(gt.shape)

    # normal_likelihood, abnormal_likelihood, z_ll, z_abnormal = nf_test_epoch(test_loader, normalizing_flow, pos_embed,
    #                                                                          gt, 'ucf-crime', 0, args)

    # print(z_ll, z_max, z_min)
    normal_samples = DataLoader(get_normal_samples(test_loader, gt), batch_size=16, shuffle=False)
    lls = []
    for normal_sample in tqdm(normal_samples):
        pos = pos_embed(normal_sample.unsqueeze(0)).squeeze()
        with torch.no_grad():
            z, log_jat = normalizing_flow(normal_sample, [pos, ])

        ll = (get_logp(1024, z, log_jat) / 1024).detach().cpu().numpy()
        lls.extend(ll.tolist())
    df = pd.DataFrame({
        'll': lls
    })
    print(df.describe())
    exit()
    gaussian_samples = DataLoader(multi_gaussian.sample(sample_shape=(200000,)), batch_size=16)

    lls = []
    for gaussian_sample in tqdm(gaussian_samples):
        pos = pos_embed(gaussian_sample.unsqueeze(0)).squeeze()
        normal_sample_, log_jat_ = normalizing_flow(gaussian_sample, [pos, ], rev=True)
        pos_ = pos_embed(normal_sample_.unsqueeze(0)).squeeze()
        z, log_jat = normalizing_flow(normal_sample_, [pos_, ])
        ll = (get_logp(1024, z, log_jat) / 1024).detach().cpu().numpy()

    print(df.describe())
    exit()
    generated_sample, _ = normalizing_flow(z, [pos, ], rev=True)

    pos = pos_embed(generated_sample.unsqueeze(0)).squeeze()
    z_, log_jat_ = normalizing_flow(generated_sample, ([pos, ]))
    print(generated_sample.shape)
    logp = get_logp(1024, z_, log_jat_) / 1024
    print(torch.argmax(logp), torch.topk(logp, k=10).values)

    sample = generated_sample[torch.argmax(logp)]
    print(torch.cosine_similarity(sample[None, :], normal_sample, dim=-1))

    # print((get_logp(1024, z_, log_jat_) / 1024).max())
    # cosin = torch.cosine_similarity(generated_sample.unsqueeze(1), normal_sample.unsqueeze(0), dim=-1)
    # print(cosin.max(dim=0))
