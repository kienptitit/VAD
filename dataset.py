from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import os
from config import CFG


def equal_samples(data_normal, data_abnormal, mode=1):
    if mode == 1:
        N = len(data_normal)
        A = len(data_abnormal)
        if N > A:
            c = N // A
            r = max(0, N % A)
            data_abnormal = data_abnormal.repeat(c, 1, 1)
            random_idx = torch.randint(0, A, (r,))
            data_abnormal = torch.concat([data_abnormal, data_abnormal[random_idx]], dim=0)  # [N,16,1024]
        else:
            c = A // N
            r = max(0, A % N)
            data_normal = data_normal.repeat(c, 1, 1)
            random_idx = torch.randint(0, N, (r,))
            data_normal = torch.concat([data_normal, data_normal[random_idx]], dim=0)  # [N,16,1024]
        return data_normal, data_abnormal
    else:
        N = len(data_normal)
        A = len(data_abnormal)
        if N > A:
            c = N // A
            r = max(0, N % A)
            data_abnormal = np.repeat(data_abnormal, c)
            random_samples = np.array([data_abnormal[random.randint(0, A - 1)] for _ in range(r)])
            data_abnormal = np.concatenate([data_abnormal, random_samples])
        else:
            c = A // N
            r = max(0, A % N)
            data_normal = np.repeat(data_normal, c)
            random_samples = np.array([data_normal[random.randint(0, N - 1)] for _ in range(r)])
            data_normal = np.concatenate([data_normal, random_samples])
        return data_normal, data_abnormal


def mycollate(batch):
    b, n_crop, t, c = batch[0].shape
    batch = torch.from_numpy(np.concatenate(batch, axis=1))
    batch = batch.reshape(-1, n_crop, t, c)

    return batch


def get_dataloader(args: CFG, mode='Normal', backbone='rtfm'):
    if mode == 'Normal':
        data_train = torch.from_numpy(np.load(args.train_path)).reshape(-1, args.snippets, 1024)
        label_train = torch.load(args.label_train_path)
        dataset_train = MyDataset(data_train, label_train)
        train_loader = DataLoader(dataset_train, batch_size=args.Batch_size, shuffle=True)

        data_test = torch.from_numpy(np.load(args.test_path)).reshape(-1, args.snippets, 1024)

        label_test = None
        dataset_test = MyDataset(data_test, label_test, mode='test')
        test_loader = DataLoader(dataset_test, batch_size=args.Batch_size, shuffle=False)

        return train_loader, test_loader
    else:
        train_path = args.train_path_MGFN
        test_path = args.test_path_MGFN
        train_data = MyDataset10Crop(train_path, backbone=backbone)
        test_data = MyDataset10Crop(test_path, mode='test', backbone=backbone)
        train_loader = DataLoader(train_data, batch_size=args.Batch_size, shuffle=True,
                                  collate_fn=mycollate)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        return train_loader, test_loader


class MyDataset10Crop(Dataset):
    def __init__(self, folder_path, mode='train', backbone='rtfm', MIL=False):
        self.path_list = np.array(sorted(os.listdir(folder_path)))
        label = np.array([0 if 'Normal' in f else 1 for f in self.path_list])
        self.nomral_path_list = self.path_list[label == 0]
        self.abnomral_path_list = self.path_list[label == 1]
        self.nomral_path_list, self.abnomral_path_list = equal_samples(self.nomral_path_list, self.abnomral_path_list,
                                                                       mode=2)
        self.folder_path = folder_path
        self.mode = mode
        self.backbone = backbone
        self.MIL = MIL

    def __len__(self):
        if self.mode == 'train':
            return min(len(self.nomral_path_list), len(self.abnomral_path_list))
        else:
            return len(self.path_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            normal = np.load(os.path.join(self.folder_path, self.nomral_path_list[idx])).transpose(1, 0, 2)
            abnormal = np.load(os.path.join(self.folder_path, self.nomral_path_list[idx])).transpose(1, 0, 2)

            new_normal = []
            new_abnormal = []

            for n, a in zip(normal, abnormal):
                feature_n = n
                feature_a = a
                if self.backbone == 'mgfn':
                    feature_n = np.concatenate([feature_n, np.linalg.norm(feature_n, axis=-1, keepdims=True)], axis=1)
                    feature_a = np.concatenate([feature_a, np.linalg.norm(feature_a, axis=-1, keepdims=True)], axis=1)
                new_normal.append(feature_n)
                new_abnormal.append(feature_a)
            new_normal = np.stack(new_normal)  # 10,32,1025
            new_abnormal = np.stack(new_abnormal)  # 10,32,1025
            if not self.MIL:
                return np.stack([new_normal, new_abnormal])
            else:
                return np.concatenate([new_normal, new_abnormal], axis=1).mean(axis=0)
        else:
            features = np.load(os.path.join(self.folder_path, self.path_list[idx])).transpose(1, 0, 2)
            if not self.MIL:
                return features
            else:
                return features.mean(axis=0)
