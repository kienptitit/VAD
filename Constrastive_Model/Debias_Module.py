import torch
import torch.nn as nn
import torch.nn.functional as F
from MGFN_Backbone import mgfn
from RTFM_Backbone import rtfm
import torch.nn.functional as F
from NormalizingFlow.model import *
from config import CFG


class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, dropout_prob=0.5):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.init_param()

    def init_param(self):
        for param in self.parameters():
            if isinstance(param, nn.Linear):
                torch.nn.init.xavier_uniform_(param.weight)

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x1 = self.fc2(x)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.dropout2(x1)

        x2 = self.fc3(x1)

        return x2


# TODO : Implement Debias Loss
class Debias_Loss(nn.Module):
    def __init__(self, tau_plus, temperature, mode="one_positive", debias=True):
        super(Debias_Loss, self).__init__()
        self.tau_plus = tau_plus
        self.temperature = temperature
        self.mode = mode
        self.debias = debias

    def forward(self, anchor, positive, negative):
        """
        :param anchor: [N,1024]
        :param positive: [N,1024]
        :param negative: [N,1024]
        :return:
        """
        # Compute Cosin Positive

        positive_cosin = torch.exp(F.cosine_similarity(anchor, positive, dim=-1) / (self.temperatur))  # N

        negative_cosin = torch.exp(F.cosine_similarity(anchor.unsqueeze(1), negative.unsqueeze(0), dim=-1) / (
            self.temperature))

        if self.debias:
            N = negative.shape[0]
            Ng = (-self.tau_plus * N * positive_cosin + negative_cosin.sum(dim=-1)) / (1 - self.tau_plus)
            Ng = torch.clamp(Ng, min=N * torch.e ** (-1 / (self.temperature)))
        else:
            Ng = negative_cosin.sum(dim=-1)
        if torch.isinf((- torch.log(positive_cosin / (positive_cosin + Ng))).mean()) or torch.isnan(
                (- torch.log(positive_cosin / (positive_cosin + Ng))).mean()):
            print((- torch.log(positive_cosin / (positive_cosin + Ng))).mean())
            print(anchor)
            print(positive)
            print((anchor * positive).sum(dim=-1) / (self.temperature))
            exit()
        p_cosin = F.cosine_similarity(anchor, positive, dim=-1)
        n_cosin = F.cosine_similarity(anchor.unsqueeze(1), negative.unsqueeze(0), dim=-1)
        return (- torch.log(positive_cosin / (positive_cosin + Ng))).mean(), p_cosin, n_cosin, positive_cosin + Ng


# TODO : Implement Sampling From Multivariate Gaussian
def Sampling(dimension=1024, num_samples=32):
    mean = torch.zeros(dimension)
    covariance_matrix = torch.eye(dimension)
    multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    samples = multivariate_normal.sample((num_samples,))
    return samples


# TODO: Implement Class Prtrain_Module
class Pretrain_Module(nn.Module):
    def __init__(self, normalizing_flow,
                 positional_embedding, device='cuda', mode='1-crop', backbone='rtfm'):
        super(Pretrain_Module, self).__init__()
        self.normalizing_flow = normalizing_flow
        self.normalizing_flow.eval()
        self.positional_embedding = positional_embedding
        self.device = device
        if backbone == 'mgfn':
            self.bb = mgfn()
            self.mlp = MLP(256, 256, 128)
        elif backbone == 'rtfm':
            self.bb = rtfm(1024)
            self.mlp = MLP(1024, 512, 512)

        self.mode = mode
        self.backbone = backbone

    def forward(self, inp):
        """
        :param inp: [16,10,64,1024]
        :return:
        """
        inp = inp.mean(dim=1)
        b_twice, n_crops, c = inp.shape
        normal_samples = inp[:b_twice // 2, :, :]  # 16,32,1024
        abnormal_samples = inp[b_twice // 2:, :, :]  # 16,32,1024

        if self.mode == '1-crop':
            c = c - 1 if self.backbone == 'mgfn' else c
            b = normal_samples.shape[0]
            positive_samples = Sampling(dimension=c, num_samples=b * n_crops).to(self.device)  # 16 * 32, 1024
            pos = self.positional_embedding(normal_samples).reshape(b * n_crops, -1)

        with torch.no_grad():
            positive_samples_original_space, _ = self.normalizing_flow(positive_samples, [pos, ], rev=True)
            if self.backbone == 'mgfn':
                positive_samples_original_space = positive_samples_original_space.reshape(b * n_crops, -1)
                l2_norm = torch.linalg.norm(positive_samples_original_space, dim=-1, keepdim=True)
                positive_samples_original_space = torch.concat([positive_samples_original_space, l2_norm], dim=-1)
            elif self.mode == 'rtfm':
                positive_samples_original_space = positive_samples_original_space.reshape(b * n_crops, -1)

        normal_samples = self.bb(normal_samples.unsqueeze(1)).squeeze()
        abnormal_samples = self.bb(abnormal_samples.unsqueeze(1)).squeeze()

        normal_samples = normal_samples.reshape(b * n_crops, -1)
        abnormal_samples = abnormal_samples.reshape(b * n_crops, -1)
        # Pass Anchor,Positive,Negative Through MLP
        normal_samples = self.mlp(normal_samples)
        abnormal_samples = self.mlp(abnormal_samples)
        positive_samples_original_space = self.mlp(positive_samples_original_space)

        return normal_samples, positive_samples_original_space, abnormal_samples


if __name__ == '__main__':
    args = CFG()

    pos_embed = PositionalEncoding1D(args.pos_embed_dim)  # Correct
    normalizing_flow = get_flow_model(args, args.in_features)
    debias_loss = Debias_Loss(args.tau_plus, args.temperature)
    Pretrain_Model = Pretrain_Module(normalizing_flow, pos_embed, mode='1-crop', backbone='rtfm').to(args.device)

    inp = torch.randn(32, 10, 32, 1024).to(args.device)
    Pretrain_Model(inp)
