import argparse
import numpy as np
import os.path as osp

import random

import torch
from tqdm import trange

from prepare_data import Data
from run import run
from model import ModelV1
from utils import fix_seed
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='3')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--load_path', type=str, default='/data/anonymous123/new_subgraph1', help='data path')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--temper', type=float, default=1.,
                        help='The temperature of the difference between '
                             'alpha and threshold when calculating gate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--train_ratio', type=float, default=50, help="Training ratio (0, 100)")
    parser.add_argument('--valid_ratio', type=float, default=25, help="Validation ratio (0, 100)")
    parser.add_argument('--hidden_channels', type=int, default=8, help="hidden channels")
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--check-point', type=int, default=5, help="Check point")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--check_edge', action='store_false',
                        help='whether check edge labels')
    parser.add_argument('--early_stopping', action='store_true',
                        help='whether use early-stop')
    parser.add_argument('--no_tpe', action='store_true',
                        help='no use time encoder')
    parser.add_argument('--use_gat', action='store_true',
                        help='use gat rather than threshold-gat')
    parser.add_argument('--use_gat1', action='store_true',
                        help='use gat rather than gcn in graph0')
    parser.add_argument('--use_graph1', action='store_true',
                        help='use graph aggregating the influence of neighbors of invitees')
    parser.add_argument('--no_edge_emb', action='store_true',
                        help='no use invitation edge embedding as supervision in sequence model')
    parser.add_argument('--delete_graph2', action='store_true',
                        help='whether delete the graph 2 model')
    parser.add_argument('--use_same_graph', default=None,
                        help='use same one graph model in graph1 and graph2 modules')
    parser.add_argument('--use_gru', action='store_true',
                        help='use gru as the based sequential model')
    parser.add_argument('--loss_alpha', type=float, default=1.0,
                        help='the coefficient of edge supervision loss')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    np.seterr(divide='ignore', invalid='ignore')

    ###################################### load data ######################################
    load_path = args.load_path

    data = Data(load_path, args)

    print('building models ......')
    model = ModelV1(args, data.x.size(1), args.hidden_channels, 2, 2,
                    data.max_len, data.n, args.temper).to(device)

    test_result = []
    for s in trange(10):
        fix_seed(int(s + 40))
        result = run(data.model_path, data, model, device, args)
        test_result.append(result)
    test_result = torch.tensor(test_result)
    test_result_path = osp.join(data.model_path, 'test_result_{}.pt'.format(args.temper))
    print('test result path is {}'.format(test_result_path))
    torch.save(test_result, test_result_path)
    print(f'Final Result: {test_result.mean(axis=0)} ± {test_result.std(axis=0)}')
    print('args is {}'.format(args.__dict__))
