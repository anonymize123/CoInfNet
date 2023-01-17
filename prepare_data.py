import time
import numpy as np
import os.path as osp
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from datetime import datetime
import random
from loader import MyDataset, collate
from torch.utils.data import DataLoader
import os


class Data:
    def __init__(self, load_path, args):

        self.path = load_path
        self.args = args
        self.model_path = self.model_path_out()

        self.x = None
        self.max_len = None

        self.n = None
        self.edge_index = None
        self.edge_weight = None
        self.offline_days = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.class_weight = None
        self.process()

    def process(self):
        t = time.perf_counter()
        print('load node features ...... ')
        node_feature = np.load(osp.join(self.path, 'x.npy'), allow_pickle=True)
        y = node_feature[:, -1]
        x = node_feature[:, 1:-2]
        self.offline_days = torch.from_numpy(node_feature[:, 0]).long()
        x = MinMaxScaler().fit_transform(x)
        self.x = torch.from_numpy(x).float()
        self.n = x.shape[0]
        print('cost time is {}s'.format(time.perf_counter() - t))

        t = time.perf_counter()
        print('load edge_index ...... ')
        edge_index_weight = np.load(osp.join(self.path, 'edge_index_weight.npy'), allow_pickle=True)
        self.edge_index = torch.from_numpy(edge_index_weight[:2]).long()
        self.edge_weight = torch.log(torch.from_numpy(edge_index_weight[-1]).float() + 1)
        print('cost time is {}s'.format(time.perf_counter() - t))

        t = time.perf_counter()
        print('load invitations ...... ')
        invitations = np.load(osp.join(self.path, 'labeled_invitations_data.npy'), allow_pickle=True)
        invite_timestamp = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timestamp() for t in invitations[-3]]
        invitations[-3] = np.array(invite_timestamp) - min(invite_timestamp)
        print('cost time is {}s'.format(time.perf_counter() - t))

        invited_idxs = np.unique(invitations[1]).astype(np.int64)
        random.shuffle(invited_idxs)
        n_mask = len(invited_idxs)
        train_start, valid_start, test_start = 0, int(n_mask * 0.5), int(n_mask * 0.75)
        train_index = invited_idxs[:valid_start]
        valid_index = invited_idxs[valid_start:test_start]
        test_index = invited_idxs[test_start:]

        print('generating invitation dictionary .......')
        invitation_dict = {}
        for row, col, t, label, weight in tqdm(invitations.T):
            if col not in invitation_dict:
                invitation_dict[col] = []
            weight = np.exp(weight) - 1
            invitation_dict[col].append([row, t, label, weight])

        new_invitation_dict = {}
        max_len = -1
        for key, value in tqdm(invitation_dict.items()):
            arr = np.array(value)
            arr = np.unique(arr, axis=0)
            sort_idx = np.argsort(arr[:, 1])
            arr = arr[sort_idx]
            new_invitation_dict[key] = arr

            if arr.shape[0] > max_len:
                max_len = arr.shape[0]

        train_dataset = MyDataset(train_index, self.n, y, new_invitation_dict, max_len)
        valid_dataset = MyDataset(valid_index, self.n, y, new_invitation_dict, max_len)
        test_dataset = MyDataset(test_index, self.n, y, new_invitation_dict, max_len)

        print('creating train dataloaders ......')
        start_time = time.perf_counter()
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                       collate_fn=collate, num_workers=8, shuffle=True)
        print('Done! create train dataloaders cost time: {:.4f}'.format(time.perf_counter() - start_time))

        print('creating train dataloaders ......')
        start_time = time.perf_counter()
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size,
                                       collate_fn=collate, num_workers=4, shuffle=False)
        print('Done! create valid dataloaders cost time: {:.4f}'.format(time.perf_counter() - start_time))

        print('creating train dataloaders ......')
        start_time = time.perf_counter()
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size,
                                      collate_fn=collate, num_workers=4, shuffle=False)
        print('Done! create test dataloaders cost time: {:.4f}'.format(time.perf_counter() - start_time))

        label = y[invited_idxs]
        self.class_weight = len(label) / (2 * np.bincount(label))

    @staticmethod
    def model_path_out() -> str:

        model_path = None
        for i in range(1000):
            model_path = '/data/shangyihao/new_subgraph1/save_model/model{}'.format(i)
            if not osp.exists(model_path):
                os.makedirs(model_path)
                break
        return model_path
