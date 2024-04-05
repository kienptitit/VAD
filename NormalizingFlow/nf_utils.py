import torch
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve
from torch.distributions import MultivariateNormal

_GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))


def get_logp(C, z, logdet_J):
    p = MultivariateNormal(torch.zeros(C), torch.eye(C))

    logp = p.log_prob(z.cpu()) + logdet_J.cpu()
    return logp.cuda()


def convert_to_anomaly_scores(args, logps_list, get_train_normal_score=False, n=None):
    if isinstance(logps_list, list):
        logps = torch.cat(logps_list)  # [290,16]
    else:
        logps = logps_list  # [810,64]

    logps -= torch.max(logps)  # -inf 0
    scores = torch.exp(logps)  # 0 1

    scores = scores.max() - scores
    if get_train_normal_score:
        scores = scores.reshape(n, -1)
        if scores.shape[1] != 64:
            scores = scores.reshape(-1)
        else:
            scores = scores[:, :32].reshape(-1)
    return scores


def normal_fl_weighting(logps, gamma=0.5, alpha=11.7, normalizer=10):
    """
    Normal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer

    mask_larger = logps > -0.2
    mask_lower = logps <= -0.2
    probs = torch.exp(logps)
    fl_weights = alpha * (1 - probs).pow(gamma) * torch.abs(logps)
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_larger] = 1.0
    weights[mask_lower] = fl_weights[mask_lower]
    return weights


def abnormal_fl_weighting(logps, gamma=2, alpha=0.53, normalizer=10):
    """
    Abnormal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer
    mask_larger = logps > -1.0
    mask_lower = logps <= -1.0
    probs = torch.exp(logps)
    fl_weights = alpha * (1 + probs).pow(gamma) * (1 / torch.abs(logps))
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_lower] = 1.0
    weights[mask_larger] = fl_weights[mask_larger]

    return weights


def get_logp_boundary(logps, mask, pos_beta=0.4, margin_abnormal_negative=0.2 / 10, margin_abnormal_positive=0.1 / 10,
                      normalizer=10):
    """
    :param margin_abnormal_positive:
    :type margin_abnormal_negative: object
    :param logps: [Batch_size * 16(32)]
    :param mask: [Batch_size * 16(32)]
    :param pos_beta: ppf
    :param normalizer:
    :return:
    """
    normal_logps = logps[mask == 0].detach()

    n_idx = int(((mask == 0).sum() * pos_beta).item())

    sorted_indices = torch.sort(normal_logps)[1]

    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]  # normal boundary
    b_n = b_n / normalizer

    b_a_negative = b_n - margin_abnormal_negative  # abnormal boundary
    b_a_positive = b_n - margin_abnormal_positive
    return b_n, b_a_negative, b_a_positive


def calculate_bg_spp_loss(logps, mask, boundaries, normalizer=10, weights=None, mode=1):
    logps = logps / normalizer

    b_n = boundaries[0]
    b_a_negative = boundaries[1]
    b_a_positive = boundaries[2]
    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 2, 1)
    # sns.histplot(logps.detach().cpu().numpy(), kde=True)
    # plt.subplot(1, 2, 2)
    # sns.histplot(logps[mask == 0].detach().cpu().numpy(), kde=True, label='normal', color='blue')
    # sns.histplot(logps[mask == 1].detach().cpu().numpy(), kde=True, label='abnormal', color='red')
    # plt.legend()
    # plt.show()
    # exit()
    if mode == 1:
        normal_logps = logps[mask == 0]
        normal_logps_inter = normal_logps[normal_logps <= b_n]
        loss_n = b_n - normal_logps_inter

        abnormal_logps = logps[mask == 1]
        abnormal_logps_inter = abnormal_logps[(abnormal_logps <= b_n) & (abnormal_logps >= b_a_positive)]
        loss_a_positive = b_n - abnormal_logps_inter

        abnormal_logps_left = abnormal_logps[(abnormal_logps >= b_a_negative) & (abnormal_logps <= b_a_positive)]
        loss_a_negative = abnormal_logps_left - b_a_negative
        if weights is not None:
            nor_weights = weights[mask == 0][normal_logps <= b_n]
            ano_weights_positive = weights[mask == 1][(abnormal_logps <= b_n) & (abnormal_logps >= b_a_positive)]
            loss_n = loss_n * nor_weights
            loss_a_positive = loss_a_positive * ano_weights_positive

            ano_weights_negative = weights[mask == 1][
                (abnormal_logps >= b_a_negative) & (abnormal_logps <= b_a_positive)]
            loss_a_negative = loss_a_negative * ano_weights_negative
        # print('Debug', len(loss_a_negative), len(loss_a_negative))
        return loss_n.mean(), loss_a_positive.mean(), loss_a_negative.mean()
    else:
        b_n = boundaries[0]  # normal boundaries
        normal_logps = logps[mask == 0]

        normal_logps_inter = normal_logps[normal_logps <= b_n]
        loss_n = b_n - normal_logps_inter

        b_a = boundaries[1]
        anomaly_logps = logps[mask == 1]
        anomaly_logps_inter = anomaly_logps[anomaly_logps >= b_a]
        loss_a = anomaly_logps_inter - b_a

        if weights is not None:
            nor_weights = weights[mask == 0][normal_logps <= b_n]
            loss_n = loss_n * nor_weights
            ano_weights = weights[mask == 1][anomaly_logps >= b_a]
            loss_a = loss_a * ano_weights

        loss_n = torch.mean(loss_n) if len(loss_n) != 0 else 0
        loss_a = torch.mean(loss_a) if len(loss_a) != 0 else 0

        return loss_n, loss_a, 0
