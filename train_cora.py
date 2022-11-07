from __future__ import division
from __future__ import print_function

import math
import os, sys
import warnings

import matplotlib
import networkx as nx
from scipy import io
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments

import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch

from torch import optim
import torch.nn.functional as F
from model import LinTrans
from optimizer import loss_function
from utils import *
from sklearn.cluster import SpectralClustering, KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics


# matplotlib.use('TkAgg')


def clustering(Kmeans, feature, true_labels):

    predict_labels = Kmeans.fit_predict(feature)  #
    cm = clustering_metrics(true_labels, predict_labels)
    acc, nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)  #
    db = -metrics.davies_bouldin_score(feature, predict_labels)  #

    return db, acc, nmi, adj, predict_labels


def ClusterCenterLocation(graph, feature):
    G = nx.Graph(graph)
    n = G.number_of_nodes()

    # calculating structural centrality of each node
    f_adj = cosine_similarity(feature)

    sim = []
    for i in range(n):
        cnt_sim = 0.
        cnt = 0
        for neigh in G.neighbors(i):
            cnt_sim += f_adj[i][neigh]
            cnt += 1
        if cnt == 0:
            sim.append(0.)
        else:
            sim.append(cnt_sim / cnt)

    degree = list(dict(nx.degree(G)).values())

    sc = []
    for i in range(n):
        p = degree[i]
        min_non_s = 1
        for j in range(n):  #
            if degree[j] > p and min_non_s > f_adj[i][j]:
                min_non_s = f_adj[i][j]
        if min_non_s < 1:
            sc.append(
                p * sim[i] * (1 - min_non_s))
        else:
            sc.append(p * sim[i] * 1)

    # taking average structural centrality as threshold ε
    average_sc = 1.0 * sum(sc) / n

    candidate_center = []
    centers = []
    for i in range(n):
        if sc[i] > average_sc:
            candidate_center.append((i, sc[i]))

    print('number of nodes before filtering', len(candidate_center))

    candidate_center = sorted(candidate_center, key=lambda tup: tup[1], reverse=True)  # sorting candidate anchors in descending order

    # selecting candidate anchors
    while len(candidate_center) > 0:
        k = candidate_center.pop(0)[0]
        centers.append(k)
        del_center = []
        for i in range(len(candidate_center)):
            v = candidate_center[i][0]
            if v in G.neighbors(k) or f_adj[k][v] >= sim[k]:  # merging ajacent or similar nodes to further reduce candidate anchors
                del_center.append(candidate_center[i])
        for tup in del_center:
            candidate_center.remove(tup)

    # matching each anchor and its neighbors as postive samples
    pos_samples = []
    for center in centers:
        for node in G.neighbors(center):
            pos_samples.append((center, node))
        pos_samples.append((center, center))

    # matching each anchor and subsequent anchors and their neighbors as postive samples
    neg_samples = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            neg_samples.append((centers[i], centers[j]))
            for node in G.neighbors(centers[j]):
                if sim[centers[j]] < f_adj[centers[j]][node]:
                    neg_samples.append((centers[i], node))

    print('number of anchors:', len(centers), 'number of positive samples：', len(pos_samples), 'number of negative samples', len(neg_samples))
    return pos_samples, neg_samples


def gae_for(args):
    print("Using {} dataset".format(args.dataset))
    if args.dataset == 'cora':
        n_clusters = 7
        Kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    elif args.dataset == 'citeseer':
        n_clusters = 6
        Kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    elif args.dataset == 'pubmed':
        n_clusters = 3
        Kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    elif args.dataset == 'wiki':
        n_clusters = 17
        Kmeans = KMeans(n_clusters=n_clusters, n_init=20)

    adj, features, true_labels, _, _, _ = load_data(args.dataset)

    n_nodes, feat_dim = features.shape
    dims = [feat_dim] + args.dims

    layers = args.linlayers

    # Store original adjacency matrix (without diagonal entries) for later
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # remove the self-loop of graph data before laplacian smoothing

    adj.eliminate_zeros()

    print('cluster center locating')
    pos_samples, neg_samples = ClusterCenterLocation(adj, features)
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)  # constructing laplacian smoothing filter
    features = sp.csr_matrix(features).toarray()

    best_embedding = features
    # np.save(f'raw-{args.dataset}.npy', best_embedding)
    print('Laplacian Smoothing...')
    for a in adj_norm_s:
        features = a.dot(features)  # X^t = H*X^(t-1)

    db, best_acc, best_nmi, best_adj, _ = clustering(Kmeans, features, true_labels)  #
    # print('db: {}, acc: {}, nmi: {}, adj: {}'.format(db, best_acc, best_nmi, best_adj))
    best_cl = db
    model = LinTrans(layers, dims)  # 线性编码器

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 5e-4

    features = torch.FloatTensor(features)

    if args.cuda:
        model.cuda()
        inx = features.cuda()

    print('Start Training...')
    best_epoch = 0

    for epoch in tqdm(range(args.epochs)):
        # balancing the number of positive samples and negative samples
        indices = np.array(len(neg_samples))
        # sample_indices = torch.LongTensor(np.random.choice(indices, size=len(pos_samples), replace=False))
        sample_indices = torch.LongTensor(np.random.choice(indices, size=len(pos_samples), replace=False))
        pos_samples_tensor = torch.LongTensor(pos_samples)
        neg_samples_tensor = torch.LongTensor(neg_samples)
        sampled_inds = torch.cat((pos_samples_tensor, neg_samples_tensor[sample_indices]), 0)
        indx = sampled_inds[:, 0].cuda()
        indy = sampled_inds[:, 1].cuda()

        model.train()
        t = time.time()
        optimizer.zero_grad()
        z = model(inx)

        batch_label = torch.cat((torch.ones(len(pos_samples_tensor)), torch.zeros(len(pos_samples_tensor)))).cuda()
        # batch_pred = model.dcs(zx, zy)
        batch_pred = model.dcs(z[indx], z[indy])
        loss = loss_function(adj_preds=batch_pred, adj_labels=batch_label)
        loss.backward()
        total_loss = loss.item()
        optimizer.step()

        # tqdm.write("Epoch: {}, train_loss_gae={:.5f}, cross_loss={:.5f}, csd_loss={:.5f}, time={:.5f}".format(epoch + 1,
        #                                                                                                       total_loss,
        #                                                                                                       cro_loss,
        #                                                                                                       c_loss,
        #                                                                                                       time.time() - t))
        if (epoch + 1) % args.interval == 0:
            model.eval()
            mu = model(inx).cpu().data.numpy()
            db, acc, nmi, adjscore, predict_labels = clustering(Kmeans, mu, true_labels)
            tqdm.write('epoc:{}, db: {}, acc: {}, nmi: {}, adj: {}'.format(epoch + 1, db, acc, nmi, adjscore))
            if db > best_cl:  # debiasing based on embeddings with best DBI
                best_epoch = epoch + 1
                best_cl = db
                best_acc = acc
                best_nmi = nmi
                best_adj = adjscore
                best_embedding = mu
                # debiasing false positive samples
                pos_samples_tmp = []
                for pos_pair in pos_samples:
                    if predict_labels[pos_pair[0]] == predict_labels[pos_pair[1]]:
                        pos_samples_tmp.append(pos_pair)
                pos_samples = pos_samples_tmp

                # debiasing false negetive samples
                neg_samples_tmp = []
                for neg_pair in neg_samples:
                    if predict_labels[neg_pair[0]] != predict_labels[neg_pair[1]]:
                        neg_samples_tmp.append(neg_pair)
                neg_samples = neg_samples_tmp

                print('number of positive samples：', len(pos_samples), 'number of negative samples', len(neg_samples))

    tqdm.write("Optimization Finished!")
    tqdm.write(
        'best_epoch:{}, best_db:{:.3f}, best_acc: {:.3f}, best_nmi: {:.3f}, best_adj: {:.3f}'.format(best_epoch,
                                                                                                     best_cl, best_acc,
                                                                                                     best_nmi,
                                                                                                    best_adj))

    return best_acc, best_nmi, best_adj, best_embedding


if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnnlayers', type=int, default=2, help="Number of gnn layers")
    parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')

    parser.add_argument('--dims', type=int, default=[250],
                        help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--interval', type=int, default=20,
                        help='Number of units in hidden layer 1.')

    parser.add_argument('--improve', type=bool, default=True, help='improve Graph.')
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda is True:
        print('Using GPU')
        torch.cuda.manual_seed(SEED)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args.dataset = 'cora'
    args.epochs = 300
    args.dims = [250]
    args.lr = 0.004
    args.weight_decay = 5e-4
    args.interval = 10
    args.gnnlayers = 2

    # args.dataset = 'citeseer'
    # args.epochs = 600
    # args.dims = [250]
    # args.lr = 0.008
    # args.weight_decay = 5e-4
    # args.interval = 40
    # args.gnnlayers = 2

    # args.dataset = 'pubmed'
    # args.epochs = 2000
    # args.dims = [250]
    # args.lr = 0.01
    # args.weight_decay = 5e-4
    # args.interval = 250
    # args.gnnlayers = 2
    acc, nmi, ari, best_embedding = gae_for(args)
    print(
        f'dataset:{args.dataset},epochs:{args.epochs},dims:{args.dims},lr:{args.lr},weight_decay:{args.weight_decay},interval:{args.interval},gnnlayers:{args.gnnlayers}')
    # np.save(f'GE-S-{args.dataset}.npy', best_embedding)
    #np.save(f'GE-S-D-{args.dataset}.npy', best_embedding)

