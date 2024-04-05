import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import *
import os
from sklearn import metrics
import torch.nn as nn
from dataset import *
from tqdm import tqdm


def MIL(y_pred, batch_size, is_transformer=0):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
        anomaly_index = torch.randperm(30).cuda()
        normal_index = torch.randperm(30).cuda()

        y_anomaly = y_pred[i, :32][anomaly_index]
        y_normal = y_pred[i, 32:][normal_index]

        y_anomaly_max = torch.max(y_anomaly)  # anomaly
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal)  # normal
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1. - y_anomaly_max + y_normal_max)

        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i, :31] - y_pred[i, 1:32]) ** 2) * 0.00008
    loss = (loss + sparsity + smooth) / batch_size

    return loss


class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = 0.6
        self.weight_init()
        self.vars = nn.ParameterList()

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4], vars[5])
        return torch.sigmoid(x)

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


args = CFG()
train_path = args.train_path_MGFN
test_path = args.test_path_MGFN
train_data = MyDataset10Crop(train_path, mode='train', backbone='rtfm', MIL=True)
test_data = MyDataset10Crop(test_path, mode='test', backbone='rtfm', MIL=True)

train_loader = DataLoader(train_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Learner(input_dim=1024, drop_p=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL
gt = np.load(args.ucf_gt)


def test(epoch):
    with torch.no_grad():
        model.eval()

        preds = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()

        for i, v_input in tqdm(enumerate(test_loader)):
            v_input = v_input.float().cuda(non_blocking=True).squeeze()
            # seq_len = v_input.shape[1]
            logits = model(v_input).squeeze()

            logits = torch.from_numpy(np.repeat(logits.detach().cpu().tolist(), 16)).cuda()

            # labels = gt_tmp[:seq_len * 16]
            preds = torch.concat([preds, logits])

        fpr, tpr, threshold = metrics.roc_curve(list(gt), preds.detach().cpu().numpy().tolist())
        rec_auc = metrics.auc(fpr, tpr)
        print(rec_auc)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs) in enumerate(train_loader):
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('loss = {}', train_loss / len(train_loader))
    scheduler.step()


for epoch in range(0, 75):
    train(epoch)
    test(epoch)
