import torch

from config import CFG
from NormalizingFlow.model import *
from dataset import get_dataloader
from Debias_Module import Pretrain_Module, Debias_Loss
import os
import random
from torch.optim import lr_scheduler
from datetime import datetime
import logging


def train_debias_model(args, epoch, pretrain_module, criterion, dataloader, optimizer):
    losses = 0.0
    pretrain_module.train()
    for step, loader in enumerate(dataloader):
        loader = loader.to(args.device)

        normal_samples, positive_samples_original_space, abnormal_samples = pretrain_module(loader)

        loss, positive_cosin, negative_cosin, denominator = criterion(positive_samples_original_space, normal_samples,
                                                                      abnormal_samples)

        optimizer.zero_grad()
        losses += loss.item()
        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(
                f"Training MLP Epoch:{epoch} \t Step:{step} \t Debias_Loss:{loss:.4f} \t Positive_Cosin:{positive_cosin.mean():.4f} \t Negative_Cosin:{negative_cosin.mean():.4f} \t Denominator : {denominator.mean():.4f}")
            logging.info(
                f"Training MLP Epoch:{epoch} \t Step:{step} \t Debias_Loss:{loss:.4f} \t Positive_Cosin:{positive_cosin.mean():.4f} \t Negative_Cosin:{negative_cosin.mean():.4f} \t Denominator : {denominator.mean():.4f} \n")
    print(f"Done Training MLP Epoch:{epoch}\tAvg_Debias_Loss:{losses / len(dataloader):.4f}")
    logging.info(f"Done Training MLP Epoch:{epoch}\tAvg_Debias_Loss:{losses / len(dataloader):.4f}\n")
    return losses / len(dataloader)


def get_pretrained_test_features(model, test_loader, gt):
    gt_tmp = torch.tensor(gt.copy()).cuda()
    normal_samples_bb = []
    abnormal_samples_bb = []

    normal_samples_mlp = []
    abnormal_samples_mlp = []

    for i, (v_input) in enumerate(test_loader):
        n_snippet = v_input.shape[-2]
        if n_snippet > 1000:
            continue
        v_input = v_input.float().cuda(non_blocking=True).mean(dim=1, keepdims=True)
        with torch.no_grad():
            v_input_bb = model.bb(v_input).squeeze()

            v_input_mlp = model.mlp(v_input_bb)

        v_input = v_input.squeeze(0).repeat(1, 16, 1).reshape(-1, 1024)
        v_input_bb = v_input_bb.unsqueeze(1).repeat(1, 16, 1).reshape(-1, 1024)
        v_input_mlp = v_input_mlp.unsqueeze(1).repeat(1, 16, 1).reshape(-1, 128)
        labels = gt_tmp[: n_snippet * 16]

        # normal_samples_bb.append(v_input_bb[labels == 0].unique(dim=0))
        # normal_samples_mlp.append(v_input_mlp[labels == 0].unique(dim=0))
        normal = v_input[labels == 0].unique(dim=0)
        normal_bb = v_input_bb[labels == 0].unique(dim=0)
        normal_mlp = v_input_mlp[labels == 0].unique(dim=0)

        if len(normal) == 0:
            continue

        if 1 in labels:
            abnormal = v_input[labels == 1].unique(dim=0)
            abnormal_bb = v_input_bb[labels == 1].unique(dim=0)
            abnormal_mlp = v_input_mlp[labels == 1].unique(dim=0)

            abnormal_cosin, normal_cosin = compute_cosin(normal, abnormal)
            abnormal_cosin_bb, normal_cosin_bb = compute_cosin(normal_bb, abnormal_bb)
            abnormal_cosin_mlp, normal_cosin_mlp = compute_cosin(normal_mlp, abnormal_mlp)
            print(
                f"Video-{i}th \t Abnormal_Cosin_BB: {abnormal_cosin_bb:.4f} \t Normal_Cosin_BB: {normal_cosin_bb:.4f} \t "
                f"Abnormal_Cosin_MLP: {abnormal_cosin_mlp:.4f} \t Normal_Cosin_MLP: {normal_cosin_mlp:.4f} \t "
                f"Abnormal_Cosin_Original: {abnormal_cosin:.4f} \t Normal_Cosin_Original: {normal_cosin:.4f}")
            logging.info(
                f"Video-{i}th \t Abnormal_Cosin_BB: {abnormal_cosin_bb:.4f} \t Normal_Cosin_BB: {normal_cosin_bb:.4f} \t "
                f"Abnormal_Cosin_MLP: {abnormal_cosin_mlp:.4f} \t Normal_Cosin_MLP: {normal_cosin_mlp:.4f} \t "
                f"Abnormal_Cosin_Original: {abnormal_cosin:.4f} \t Normal_Cosin_Original: {normal_cosin:.4f}\n")

        else:
            normal_cosin_ori = compute_cosin(normal, None)
            normal_cosin_bb = compute_cosin(normal_bb, None)
            normal_cosin_mlp = compute_cosin(normal_mlp, None)
            print(
                f"Video-{i}th \t Normal_Cosin_BB: {normal_cosin_bb:.4f} \t Normal_Cosin_MLP: {normal_cosin_mlp:.4f} \t"
                f"Normal_Cosin_Original: {normal_cosin_ori:.4f}")
            logging.info(
                f"Video-{i}th \t Normal_Cosin_BB: {normal_cosin_bb:.4f} \t Normal_Cosin_MLP: {normal_cosin_mlp:.4f} \t"
                f"Normal_Cosin_Original: {normal_cosin_ori:.4f}")
            # abnormal_samples_bb.append(v_input_bb[labels == 1].unique(dim=0))
            # abnormal_samples_mlp.append(v_input_mlp[labels == 1].unique(dim=0))

        gt_tmp = gt_tmp[n_snippet * 16:]

    # normal_bb, abnormal_bb = torch.concat(normal_samples_bb, dim=0), torch.concat(abnormal_samples_bb, dim=0)
    # normal_mlp, abnormal_mlp = torch.concat(normal_samples_mlp, dim=0), torch.concat(abnormal_samples_mlp, dim=0)

    # return normal_bb, abnormal_bb, normal_mlp, abnormal_mlp


def compute_cosin(normal, abnormal):
    normal_cosin = 0.0
    abnormal_cosin = 0.0
    for n in normal:
        if isinstance(abnormal, torch.Tensor):
            abnormal_cosin += torch.cosine_similarity(n, abnormal, dim=-1).mean()
        normal_cosin_matrix = torch.cosine_similarity(n, normal, dim=-1)
        normal_cosin += normal_cosin_matrix[normal_cosin_matrix != 1.0].mean()

    return (abnormal_cosin / len(normal), normal_cosin / len(normal)) if isinstance(abnormal,
                                                                                    torch.Tensor) else normal_cosin / len(
        normal)


def test_debias_model(model, test_loader, gt):
    model.eval()
    get_pretrained_test_features(model, test_loader, gt)

    # abnormal_cosin_bb, normal_cosin_bb = compute_cosin(normal_bb, abnormal_bb)
    # abnormal_cosin_mlp, normal_cosin_mlp = compute_cosin(normal_mlp, abnormal_mlp)

    # return abnormal_cosin_bb, normal_cosin_bb, abnormal_cosin_mlp, normal_cosin_mlp


def seed_everything():
    torch.manual_seed(22)
    torch.cuda.manual_seed(22)
    np.random.seed(22)
    random.seed(22)


def train_and_validation(args: CFG, train_loader, test_loader, Pretrain_Model, criterion,
                         mode='rtfm', writer=None):
    # Get Config
    total_epochs = args.total_constrastive_epochs
    # Get Optimizer
    optim_MLP = torch.optim.Adam(Pretrain_Model.parameters(), lr=args.MLP_learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optim_MLP, T_max=30)

    gt = np.load(args.ucf_gt)

    # TODO: Define LearningRate Scheduler
    # Training

    cosins_list = []
    cosins_backbone_list = []
    for epoch in range(total_epochs):
        _ = train_debias_model(args, epoch, Pretrain_Model, criterion, train_loader, optim_MLP)
        test_debias_model(Pretrain_Model, test_loader, gt)
        # print(
        #     f"Test Epoch {epoch} \t Abnormal_Cosin_BB: {abnormal_cosin_bb:.4f} \t Normal_Cosin_BB: {normal_cosin_bb:.4f} \t "
        #     f"Abnormal_Cosin_MLP: {abnormal_cosin_mlp:.4f} \t Normal_Cosin_MLP: {normal_cosin_mlp:.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"****************Current learning rate: {current_lr:.5f}****************")
        logging.info(f"****************Current learning rate: {current_lr:.5f}****************")

    # torch.save(Pretrain_Model.backbone.state_dict(),
    #            os.path.join(args.save_dir, 'backbone_version10.pt'))
    # torch.save(Pretrain_Model.mlp.state_dict(),
    #            os.path.join(args.save_dir, 'mlp_version10.pt'))
    # with open(os.path.join(args.save_dir, 'cosins_list_version10.pkl'), 'wb') as file:
    #     pickle.dump(cosins_list, file)
    # with open(os.path.join(args.save_dir, 'cosins_backbone_list_version10.pkl'), 'wb') as file:
    #     pickle.dump(cosins_list, file)
    return cosins_list


def main():
    seed_everything()
    args = CFG()
    save_dir = args.log_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    t = datetime.now()
    formatted_time = t.strftime('%H_%M_%S')

    file_path = os.path.join(save_dir, formatted_time + '.log')
    logging.basicConfig(filename=os.path.join(save_dir, file_path), level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%H:%M:%S')

    # Define Model
    pos_embed = PositionalEncoding1D(args.pos_embed_dim)  # Correct
    normalizing_flow = get_flow_model(args, args.in_features)
    debias_loss = Debias_Loss(args.tau_plus, args.temperature)
    Pretrain_Model = Pretrain_Module(normalizing_flow, pos_embed, mode='1-crop', backbone='rtfm').to(args.device)

    Pretrain_Model.normalizing_flow.load_state_dict(
        torch.load("/home/naver/Documents/Kien/VAD/NF_Constrastive/NormalizingFlow/nf.pt"))
    # Get DataLoader
    Pretrain_Model = Pretrain_Model.to(args.device)
    train_loader, test_loader = get_dataloader(args, mode='MGFN', backbone='rtfm')

    # Training
    cosins_list = train_and_validation(args, train_loader, test_loader, Pretrain_Model, debias_loss,
                                       mode='rtfm')
    logging.shutdown()
    return cosins_list, Pretrain_Model


if __name__ == '__main__':
    cosins_list, Pretrain_Model = main()
