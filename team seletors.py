import os
import os.path as osp
import numpy as np
from tqdm import trange
import time
import random
import argparse
import scipy.sparse as sparse
import shutil


def sample_team(idx, args):

    load_path = args.load_path
    edge_index = np.load(osp.join(load_path, 'edge_index_weight.npy'))  # (3,Ne)
    n = int(max(max(edge_index[0]), max(edge_index[1])) + 1)

    print('generating sparse adj ...')
    start_time = time.perf_counter()
    adj = sparse.coo_matrix((edge_index[2], (edge_index[0], edge_index[1])), shape=(n, n))
    print('Done ! cost time {} s'.format(time.perf_counter() - start_time))

    print('获得每行非零的索引......')
    start_time = time.perf_counter()
    adj = adj.tocsr()
    neighbors_list = np.split(adj.indices, adj.indptr[1:-1])
    print('Done ! cost time {} s'.format(time.perf_counter() - start_time))

    print('获得每行非零的索引对应的intimacy......')
    start_time = time.perf_counter()
    intimacy_dict = np.split(adj.data, adj.indptr[1:-1])
    end_time = time.perf_counter()
    print('Done ! cost time {} s'.format(end_time - start_time))
    del adj

    print('starting sample fixed teams ......')
    team_dict = {}
    for i in trange(n):
        tmp = set()
        tmp.add(i)
        nei = neighbors_list[i]
        data = intimacy_dict[i]
        sort_idx = np.argsort(-data)
        nei = nei[sort_idx]
        for u in nei:
            flag = True
            for v in tmp:
                if u not in set(neighbors_list[v]) and v not in set(neighbors_list[u]):
                    flag = False
                    break
            if flag:
                tmp.add(u)
        team_dict[i] = tmp

    out_path = osp.join(load_path, 'out')
    os.mkdir(out_path)
    team_dict_path = osp.join(out_path, 'team_dict_{}.npy'.format(idx))
    np.save(team_dict_path, team_dict, allow_pickle=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input', help='input path of data')
    parser.add_argument('--data_output', help='output path of data')
    parser.add_argument('--tb_log_dir', help='output path of log')
    parser.add_argument('--load_path', type=str, default='/data/anonymous123/new_subgraph1', help='data path')
    parser.add_argument('--shuffle_times', type=int, default=20, help='the times of shuffling neighbors')
    parser.add_argument('--sample_ratio', type=float, default=0.002, help='sample ratio of target nodes')
    parser.add_argument('--num_subgraphs', type=int, default=1, help='the number of sampled subgraphs')
    args = parser.parse_args()



    sample_team(1, args.load_path, args)


if __name__ == '__main__':
    main()