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

log_theta = torch.nn.LogSigmoid()


def cal_false_alarm(gt, preds, threshold=0.5):
    preds = list(preds.cpu().detach().numpy())
    gt = list(gt.cpu().detach().numpy())

    preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far


def convert_to_anomaly_scores(logps):
    logps = logps.clone()
    logps -= torch.max(logps)  # -inf 0

    scores = torch.exp(logps)  # 0 1

    scores = scores.max() - scores

    return scores


def \
        nf_test_epoch(dataloader, model, pos_embed, gt, dataset, epoch=0, args: CFG = None):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        abnormal_preds = torch.zeros(0).cuda()
        abnormal_labels = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()

        normal_likelihood = 0.0
        abnormal_likelihood = 0.0
        z_ = 0.0
        z_abnormal = 0.0
        cnt_abnormal = 0
        z_max = -torch.inf
        z_min = torch.inf
        mg = MultivariateNormal(torch.zeros(args.in_features), torch.eye(args.in_features))
        for i, (v_input) in tqdm(enumerate(dataloader)):
            n_snippet = v_input.shape[-2]

            v_input = v_input.float().cuda(non_blocking=True).mean(dim=1)
            pos = pos_embed(v_input)
            v_input, pos = v_input.reshape(-1, 1024), pos.reshape(-1, args.pos_embed_dim)
            z, jac = model(v_input, [pos, ])
            dim = z.shape[-1]
            logps = get_logp(dim, z, jac)
            logps = logps / dim
            logits = convert_to_anomaly_scores(logps)

            pred = torch.cat((pred, logits))

            z_ll = mg.log_prob(z.cpu()) / 1024
            labels = gt_tmp[: n_snippet * 16]
            pred_tmp = np.repeat(list(logits.detach().cpu().numpy()), 16)

            z_tmp = np.repeat(list(z_ll.detach().cpu().numpy()), 16)
            logps_tmp = np.repeat(list(logps.detach().cpu().numpy()), 16)
            normal_likelihood += logps_tmp[labels.cpu().numpy() == 0].mean()
            z_ += z_tmp[labels.cpu().numpy() == 0].mean()
            if 1 in labels:
                abnormal_likelihood += logps_tmp[labels.cpu().numpy() == 1].mean()
                cnt_abnormal += 1
            z_max = max(z_max, z_tmp[labels.cpu().numpy() == 0].max())
            z_min = min(z_min, z_tmp[labels.cpu().numpy() == 0].min())
            # abnormal_likelihood += logps_tmp[labels.cpu().numpy() == 1].mean()

            figure_path = os.path.join(args.nf_save_score, f'video_{i}-th')
            if not os.path.exists(figure_path):
                os.mkdir(figure_path)
            plt.figure()
            plt.plot(logps_tmp, label='Pred')
            plt.plot(labels.detach().cpu().numpy(), label='GT')
            plt.legend()
            plt.savefig(os.path.join(figure_path, f"Epoch_{epoch}.jpg"))
            plt.close()

            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            else:
                abnormal_labels = torch.cat((abnormal_labels, labels))
                abnormal_preds = torch.cat((abnormal_preds, logits))
            gt_tmp = gt_tmp[n_snippet * 16:]
        z_abnormal /= cnt_abnormal
        normal_likelihood /= len(dataloader)
        abnormal_likelihood /= len(dataloader)
        z_ /= len(dataloader)
        pred = list(pred.cpu().detach().numpy())
        n_far = 0.0
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))

        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)

        if dataset == 'ucf-crime':
            return normal_likelihood, abnormal_likelihood, z_, z_abnormal
        elif dataset == 'xd-violence':
            return pr_auc, n_far
        elif dataset == 'shanghaiTech':
            return roc_auc, n_far
        else:
            raise RuntimeError('Invalid dataset.')


def nf_train_epoch(args, epoch, normal_only, normalizing_flow, pos_embedding, dataloader, optimizer):
    total_loss = 0.0
    total_loss_abnormal = 0.0
    total_loss_normal = 0.0
    mg = MultivariateNormal(torch.zeros(args.in_features), torch.eye(args.in_features))
    total_zll = 0.0
    for step, loader in enumerate(dataloader):
        """
        loader : [B * 2,10,200,1024]
        """

        # Define Some Variables

        # loader = torch.concat(torch.split(loader, loader.shape[0] // 2, dim=0), dim=2)[:, args.index,
        #          :]  # [B,10,400,1024]
        loader = torch.concat(torch.split(loader, loader.shape[0] // 2, dim=0), dim=2)  # [B,10,400,1024]
        loader = loader.mean(dim=1)

        # loader = loader[:,args.index,:,:]  # B,400,1024

        m_b = torch.hstack([torch.zeros(loader.shape[1] // 2), torch.ones(loader.shape[1] // 2)]).unsqueeze(
            0).repeat(loader.shape[0], 1)  # [B,Snippet * 2]

        if normal_only:
            loader = loader[:, :args.snippets, :]
            m_b = m_b[:, :args.snippets]

        b, n, c = loader.shape
        loader = loader.reshape(-1, c)  # [B * Snippet , 1024]
        m_b = m_b.reshape(-1)  # [B * Snippet]
        pos_embed = pos_embedding(torch.rand(b, args.snippets, c)).reshape(-1,
                                                                           args.pos_embed_dim)  # [B * Snippet , 128]

        e_b = loader.clone()
        m_b = m_b
        p_b = pos_embed
        # Forward Through Normalizing FLows
        if normal_only:
            e_b = e_b.to(args.device)
            p_b = p_b.to(args.device)

            if args.flow_arch == 'flow_model':
                z, log_jac_det = normalizing_flow(e_b)  # [4*16,1024] , [4*16]

            else:
                z, log_jac_det = normalizing_flow(e_b, [p_b, ])
                z_ll = mg.log_prob(z.cpu()) / c
                total_zll += z_ll.mean()
                # logps = get_logp(c, z, log_jac_det)
                # logps /= c
                #
                # lops = log_theta(logps.mean())
                # logits = convert_to_anomaly_scores(logps)
                #
                # if step % 1 == 0:
                #     print(
                #         f"Normal Video Training Score: Epoch: {epoch} \t Step: {step} \t  log_prob: {logps.detach().cpu().mean():.4f} log_theta: {lops:.4f}")

        else:
            if args.flow_arch == 'flow_model':
                z, log_jac_det = normalizing_flow(e_b)
            else:
                e_b_normal = e_b[m_b == 0]
                e_b_abnormal = e_b[m_b != 0]

                p_b = p_b.to(args.device)
                e_b_normal, e_b_abnormal = e_b_normal.to(args.device), e_b_abnormal.to(args.device)

                z_normal, log_jac_det_normal = normalizing_flow(e_b_normal, [p_b, ])
                z_abnormal, log_jac_det_abnormal = normalizing_flow(e_b_abnormal, [p_b, ])

                z = torch.concat([z_normal, z_abnormal], dim=0)
                log_jac_det = torch.hstack([log_jac_det_normal, log_jac_det_abnormal])

                m_b = torch.hstack([torch.zeros(e_b_normal.shape[0]), torch.ones(e_b_abnormal.shape[0])])

        # Compute Loss
        if normal_only:
            logps = get_logp(c, z, log_jac_det)  # Batch_size * 16
            logps = logps / c
            if args.focal_weighting:
                normal_weights = normal_fl_weighting(logps.detach())

                loss = -log_theta(logps) * normal_weights
                loss = loss.mean()
            else:
                # logps = log_theta(logps)
                # loss = -logps.mean()
                loss = -log_theta(logps.mean())
        else:
            logps = get_logp(c, z, log_jac_det)

            logps = logps / c
            if args.focal_weighting:
                logps_detach = logps.detach()
                normal_logps = logps_detach[m_b == 0]
                anomaly_logps = logps_detach[m_b == 1]
                nor_weights = normal_fl_weighting(normal_logps)
                ano_weights = abnormal_fl_weighting(anomaly_logps)
                weights = nor_weights.new_zeros(logps_detach.shape)
                weights[m_b == 0] = nor_weights
                weights[m_b == 1] = ano_weights
                loss_ml = -log_theta(logps[m_b == 0]) * nor_weights  # (256, )
                loss_ml = torch.mean(loss_ml)
            else:

                loss_ml = -log_theta(logps[m_b == 0].mean())

            boundaries = get_logp_boundary(logps, m_b, args.pos_beta, args.margin_abnormal_negative,
                                           args.margin_abnormal_positive, args.normalizer)
            if args.focal_weighting:
                loss_n_con, loss_a_con_pos, loss_a_con_neg = calculate_bg_spp_loss(logps, m_b, boundaries,
                                                                                   args.normalizer,
                                                                                   weights=weights,
                                                                                   mode=args.mode_loss)
            else:
                loss_n_con, loss_a_con_pos, loss_a_con_neg = calculate_bg_spp_loss(logps, m_b, boundaries,
                                                                                   args.normalizer,
                                                                                   mode=args.mode_loss)
            loss = loss_ml + args.bgspp_lambda * (loss_n_con + loss_a_con_pos + loss_a_con_neg)
            total_loss_abnormal += loss_a_con_pos
            total_loss_normal += loss_n_con

        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Statistic
        if step % 1 == 0:
            if normal_only:
                print(f"Epoch: {epoch} \t Step: {step} \t MLE_Loss: {loss:.4f}")
            else:
                print(
                    f"Epoch:{epoch} \t Step:{step} \t MLE_Loss:{loss:.4f} \t Loss_Normal:{loss_n_con:.4f} \t Loss_Abnormal:{loss_a_con_pos:.4f}")
    if normal_only:
        print(
            f"Done Training Normalizing Flow (Normal Only) Epoch: {epoch} \t Avg_MLE_Loss: {total_loss / len(dataloader):.4f} \t Z_LL: {total_zll / len(dataloader):.4f}")
    else:
        print(
            f"Done Training Normalizing Flow (All) Epoch: {epoch}\t"
            f"Avg_MLE_Loss: {total_loss / len(dataloader):.4f} \t Avg_Loss_Normal: {total_loss_normal / len(dataloader):.4f}"
            f" \t Avg_Loss_Abnormal: {total_loss_abnormal / len(dataloader):.4f}")


def seed_everything():
    torch.manual_seed(22)
    torch.cuda.manual_seed(22)
    np.random.seed(22)
    random.seed(22)


def train():
    seed_everything()

    args = CFG()
    normalizing_flow = get_flow_model(args, args.in_features).to(args.device)

    pos_embed = PositionalEncoding1D(args.pos_embed_dim)  # Correct
    optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    train_loader, test_loader = get_dataloader(args, mode='MGFN', backbone='mgfn')
    normal_epochs = 9
    total_epochs = 9

    gt = np.load(args.ucf_gt)

    for epoch in range(total_epochs):
        normal_only = epoch < normal_epochs
        nf_train_epoch(args, epoch, normal_only, normalizing_flow,
                       pos_embed, train_loader, optimizer)
        normal_likelihood, abnormal_likelihood, z_ll = nf_test_epoch(test_loader, normalizing_flow, pos_embed,
                                                                     gt, 'ucf-crime', epoch, args)

        print(f"Test Epoch \t {epoch} \t Normal_ll: {normal_likelihood:.4f} \t Prior: {z_ll:.4f}")
        # if epoch % 10 == 0:
        #     torch.save(normalizing_flow.state_dict(), f'normalizing_flow_31_1_{epoch}.pt')
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"****************Current learning rate: {current_lr:.5f}****************")
    torch.save(normalizing_flow.state_dict(), 'nf.pt')


if __name__ == '__main__':
    train()
