import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import re
from torch.utils.data import DataLoader,Dataset
from utility.GAT_model import *
from utility.arguments import *
import torch
import numpy as np
import os
import sys
from time import time
import random
import warnings
warnings.filterwarnings("ignore")
from utility.parser import parse_args
import scipy.sparse as sp
from utility.load_data import Data
from tqdm import tqdm

args = parse_args()
device = torch.device("cuda:%d"%args.gpu_id)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)

from utility.batch_test import data_generator, pad_sequences, words_lookup, test
from utility.model_GAT import *

doc_dict = data_generator.doc_word_list
# print(data_generator.doc_dict.keys()[0])
def normalized_adj_bi(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_adj(adj,max_length):
    # abj: [ns1xns2, ns2xns2....]
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # max_length = max([a.shape[0] for a in adj]) # a: node_size_a x node_size_a
    # mask: example_num x max_length x 1
    mask = np.zeros((max_length, 1)) # mask for padding #
    adj = adj+np.eye(len(adj))
    # for i in tqdm(range(adj.shape[0])):
    adj_normalized = normalized_adj_bi(adj) # no self-loop
    pad = max_length - adj_normalized.shape[0] # padding for each epoch
    adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
    adj = adj_normalized
    mask[:adj.shape[0],:] = 1
    return np.array(list(adj)), mask

def words_lookup(docs):
    if args.model == 'GRMM':
        return [data_generator.doc_unqiue_word_list[i] for i in docs]
    else:
        return [data_generator.doc_word_list[i] for i in docs]


class GraphDataset():

    def __init__(self,window_size):
        self.window_size = window_size
        self.word_id_map = None
        self.unique_words = None
        self.shuffle_doc_name_list = None
        self.label_list = None

    def adjacent(self, doc_nodes, doc_content_list):
        batch_adj = []
        batch_mask = []
        for doc_node, words in zip(doc_nodes, doc_content_list):
            length = len(words)

            doc_word_id_map = {}
            for j in range(len(doc_node)):
                doc_word_id_map[doc_node[j]] = j
            windows = []
            if length <= self.window_size:
                windows.append(words)
            else:
                for j in range(length - self.window_size + 1):
                    window = words[j: j + self.window_size]
                    windows.append(window)
            word_pair_count = {}
            for window in windows:
                for p in range(1, len(window)):
                    for q in range(0, p):
                        word_p = window[p]
                        word_p_id = word_p # doc_word_id_map[word_p]
                        word_q = window[q]
                        word_q_id = word_q # doc_word_id_map[word_q]
                        if word_p_id == word_q_id:
                            continue
                        word_pair_key = (word_p_id, word_q_id)
                        # word co-occurrences as weights
                        if word_pair_key in word_pair_count:
                            word_pair_count[word_pair_key] += 1.
                        else:
                            word_pair_count[word_pair_key] = 1.
                        # bi-direction
                        word_pair_key = (word_q_id, word_p_id)
                        if word_pair_key in word_pair_count:
                            word_pair_count[word_pair_key] += 1.
                        else:
                            word_pair_count[word_pair_key] = 1.
            row = []
            col = []
            weight = []
            features = []

            for key in word_pair_count:
                p = key[0]
                q = key[1]
                row.append(doc_word_id_map[p]) # p
                col.append(doc_word_id_map[q]) # q
                weight.append(word_pair_count[key])
            adj = sp.csr_matrix((weight, (row, col)), shape=(len(doc_node), len(doc_node)))
            adj,mask = preprocess_adj(adj.todense(),args.doc_len)
            batch_adj.append(adj)
            batch_mask.append(mask)
        return np.array(batch_adj), np.array(batch_mask)


if __name__ == '__main__':
    device = torch.device("cuda:%d"%args.gpu_id)
    users_to_test = list(data_generator.test_set.keys())
    GD = GraphDataset(3)
    config = dict()
    config['n_docs'] = data_generator.n_docs
    config['n_qrls'] = data_generator.n_qrls
    config['n_words'] = data_generator.n_words
    # if args.model == 'GRMM':
    #     config['docs_adj'] = np.load("../Data/{}/doc_adj.npy".format(args.dataset))
    config['idf_dict'] = np.load("../data/{}/idf.npy".format(args.dataset), allow_pickle=True).item()

    model = eval(args.model)(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.l2)
    precision = []
    ndcg = []
    best_ret = 0
    best_epoch = 0
    for epoch in range(args.epoch):
        # t0 = time()
        n_batch = 32
        total_loss = 0
        model.train()
        for idx in tqdm(range(n_batch)):
            qrls, pos_docs, neg_docs = data_generator.sample()

            unique_words_pos = words_lookup(pos_docs)
            pos_doc_list = []
            for doc in pos_docs:
                pos_doc_list.append(doc_dict[doc])
            adj_mat_pos, mask_mat_pos = GD.adjacent(unique_words_pos,pos_doc_list)

            unique_words_neg = words_lookup(neg_docs)
            neg_doc_list = []
            for doc in neg_docs:
                neg_doc_list.append(doc_dict[doc])
            adj_mat_neg, mask_mat_neg  = GD.adjacent(unique_words_neg,neg_doc_list)
            pos_docs_words = pad_sequences(words_lookup(pos_docs), maxlen=args.doc_len, value=138)
            neg_docs_words = pad_sequences(words_lookup(neg_docs), maxlen=args.doc_len, value=138)
            l = [data_generator.qrl_word_list[i] for i in qrls]
            qrls_words = pad_sequences(l, maxlen=args.qrl_len, value=138)
            qrls_words = torch.tensor(qrls_words).long().to(device)
            pos_docs_words = torch.tensor(pos_docs_words).long().to(device)
            neg_docs_words = torch.tensor(neg_docs_words).long().to(device)

            pos_scores = model(adj_mat_pos, mask_mat_pos, qrls_words, pos_docs_words, pos_docs)
            neg_scores = model(adj_mat_neg, mask_mat_neg, qrls_words, neg_docs_words, neg_docs)
            loss = torch.max(torch.zeros_like(pos_scores).float().to(device), (1 - pos_scores + neg_scores))
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.cpu().detach().numpy()
            optimizer.step()
        ret = test(model, users_to_test, model_name='GAT')
        # precision.append(ret[0])
        # ndcg.append(ret[1])
        # if args.verbose:
        print("epoch:%d"%epoch, "loss:%.4f"%(total_loss/args.batch_size))
        # if ret[1] > best_ret:
        #     best_ret = ret[1]
        #     best_epoch = epoch
            # model.save('{}.{}.f{}.best.model'.format(args.model, args.dataset, str(args.fold)))

    print("P@20:", precision[best_epoch])
    print("ndcg@20:", ndcg[best_epoch])

