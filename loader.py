from torch.utils.data import Dataset
import torch
from torch_geometric.utils import k_hop_subgraph


class MyDataset(Dataset):
    def __init__(self, target_id, n, y, invitation_dict, max_len=25):
        super(MyDataset, self).__init__()

        self.target_id = target_id

        self.edge_index = None

        self.y = torch.from_numpy(y).long()

        self.n = n

        self.invitation_dict = invitation_dict

        self.max_len = max_len

    def __getitem__(self, index):
        target_id = self.target_id[index]
        invitors, timestamps, e_label, e_weight = self.invitation_dict[target_id].T
        length = len(invitors)
        invitor_ids = torch.from_numpy(invitors).long()
        invitee_ids = torch.empty((length + 1), dtype=torch.long).fill_(target_id)
        ids = torch.empty((self.max_len + 1, 2), dtype=torch.long).fill_(self.n)  # [max_len, 2]

        ids[-(length + 1):-1, 0] = invitor_ids
        ids[-1, 0] = target_id
        ids[-(length + 1):, 1] = invitee_ids

        invite_y = self.y[target_id]
        edge_labels = torch.empty(self.max_len + 1, dtype=torch.long).fill_(-1)
        edge_labels[-(length + 1): -1] = torch.from_numpy(e_label).long()
        edge_labels[-1] = -1

        edge_weights = torch.empty(self.max_len + 1, dtype=torch.float).fill_(0)
        edge_weights[-(length + 1): -1] = torch.from_numpy(e_weight).float()
        edge_weights[-1] = 1

        return ids, invite_y, edge_labels, edge_weights

    def __len__(self):
        return len(self.target_id)


def collate(batches):
    batch_size = len(batches)
    invited_ids = []
    invite_ys = torch.empty(batch_size).long()
    edge_ys = []
    edge_ws = []
    for i, batch in enumerate(batches):
        ids, invite_y, edge_labels, edge_weights = batch
        invited_ids.append(ids)
        invite_ys[i] = invite_y
        edge_ys.append(edge_labels)
        edge_ws.append(edge_weights)
    invited_ids = torch.stack(invited_ids)
    edge_ys = torch.stack(edge_ys)
    edge_ws = torch.stack(edge_ws)
    return invited_ids, invite_ys, edge_ys, edge_ws

