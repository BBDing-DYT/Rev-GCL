import argparse
import os.path as osp
import random

import yaml
from yaml import SafeLoader
import numpy as np
import torch
from torch_geometric.utils import dropout_adj

from pGRACE.model import NewEncoder, NewGRACE
from pGRACE.functional import drop_feature
from pGRACE.eval import log_regression, MulticlassEvaluator, phi
from pGRACE.dataset import get_dataset
from utils import seed_torch
from torch.optim import lr_scheduler
def train(epoch):
    model.train()
    # current_learning_rate = optimizer.param_groups[0]['lr']
    # if epoch % 500 == 0:
    #     scheduler.step()
    #     current_learning_rate = optimizer.param_groups[0]['lr']
    #
    # if current_learning_rate < 0.000002:
    #     optimizer.param_groups[0]['lr'] = 0.000002
    #     current_learning_rate = optimizer.param_groups[0]['lr']
    # print(f'Learning rate: {current_learning_rate}')
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(data.x, drop_feature_rate_1)
    x_2 = drop_feature(data.x, drop_feature_rate_2)
    # Random Propagation
    z1 = model(x_1, edge_index_1, [1, 4])
    z2 = model(x_2, edge_index_2, [1, 4])
    loss = model.loss(z1, z2, batch_size=64 if args.dataset == 'Coauthor-Phy' or args.dataset == 'ogbn-arxiv' else None)
    loss.backward()
    optimizer.step()
    return loss.item()

#node classification
def test(final=False):
    model.eval()
    z = model(data.x, data.edge_index, [1, 1], final=True)
    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or  args.dataset == 'PubMed':
            acc = log_regression(z, dataset, evaluator, split='preloaded', num_epochs=5000, preload_split=split)['acc']
        else : acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=5000, preload_split=0)['acc']
    return acc, z

# node clustering
def test_clu(final=False):
    model.eval()
    z = model(data.x, data.edge_index, [1, 1], final=True)
    y = dataset[0].y.view(-1)
    cluster_num = len(np.unique(y))
    nmi = phi(z, y, cluster_num)
    return nmi, z

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Amazon-Photo')
    parser.add_argument('--config', type=str, default='our.yaml')
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    # seed_torch(args.seed)
    torch.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(args.seed)
    use_nni = args.config == 'nni'
    learning_rate = config['learning_rate']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    drop_scheme = config['drop_scheme']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    device = torch.device(args.device)
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = data.to(device)
    # two levels
    level_channels= [512, 512]
    # four levels
    # level_channels = [512, 512, 512, 512]
    num_subnet= 8
    save_memory= True

    split = 0
    if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or  args.dataset == 'PubMed': split = (data.train_mask, data.test_mask, data.val_mask)
    encoder = NewEncoder(dataset.num_features, level_channels, num_subnet, save_memory).to(device)

    model = NewGRACE(encoder, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    log = args.verbose.split(',')
    best_acc_dyt = 0
    best_epoch = 0
    for epoch in range(1, num_epochs + 1):
        loss = train(epoch)
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        if epoch >= 0:
            if epoch % 5 == 0:
                acc, z= test(final=True)
                # acc, z = test_clu(final=True)
                if 'eval' in log:
                    if acc >= best_acc_dyt:
                        best_acc_dyt = acc
                        best_epoch = epoch
                        x_1 = drop_feature(data.x, drop_feature_rate_1)
                        x_2 = drop_feature(data.x, drop_feature_rate_2)
                        # np.save('embedding/'+args.dataset + '_Graph_embeddingfull.npy', z.detach().cpu().numpy())
                    print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}, best_acc = {best_acc_dyt}, best_epoch = {best_epoch}')


