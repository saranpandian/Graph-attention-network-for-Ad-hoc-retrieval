import pandas as pd
import numpy as np
from utility.parser import parse_args
args = parse_args()
# df = pd.read_csv("/media/sda2/Share/saran/GRMM/data/msmarco-passage/queries.tsv",sep='\t',header=None)
# idf_dict = np.load("../data/{}/idf.npy".format(args.dataset), allow_pickle=True).item()
# print(len(idf_dict))

# import numpy as np
# from tqdm import tqdm
# import scipy.sparse as sp
# from utility.batch_test import data_generator, pad_sequences, words_lookup, test
# from utility.load_data import Data
# from utility.parser import parse_args
# args = parse_args()

# doc_dict = {}
# doc_dict_path = "../data/msmarco-passage/map.documents.txt"
# # data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
# def pad_sequences(items, maxlen, value=data_generator.n_words):
#     result = []
#     for item in items:
#         if len(item) < maxlen:
#             item = item + [value] * (maxlen - len(item))
#         if len(item) > maxlen:
#             item = item[:maxlen]
#         result.append(item)
#     return result

# def normalized_adj_bi(adj):
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = np.diag(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

# def preprocess_adj(adj,max_length):
#     # abj: [ns1xns2, ns2xns2....]
#     """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
#     # max_length = max([a.shape[0] for a in adj]) # a: node_size_a x node_size_a
#     # mask: example_num x max_length x 1
#     mask = np.zeros((max_length, 1)) # mask for padding #
#     adj = adj+np.eye(len(adj))
#     # for i in tqdm(range(adj.shape[0])):
#     adj_normalized = normalized_adj_bi(adj) # no self-loop
#     pad = max_length - adj_normalized.shape[0] # padding for each epoch
#     adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
#     adj = adj_normalized
#     return np.array(list(adj))

# def words_lookup(docs):
#     if args.model == 'GRMM':
#         return [data_generator.doc_unqiue_word_list[i] for i in docs]
#     else:
#         return [data_generator.doc_word_list[i] for i in docs]


# class GraphDataset():

#     def __init__(self,window_size):
#         self.window_size = window_size
#         self.word_id_map = None
#         self.unique_words = None
#         self.shuffle_doc_name_list = None
#         self.label_list = None

#     def adjacent(self, doc_nodes, doc_content_list):
#         batch_adj = []
#         for doc_node, words in zip(doc_nodes, doc_content_list):
#             length = len(words)

#             doc_word_id_map = {}
#             for j in range(len(doc_node)):
#                 doc_word_id_map[doc_node[j]] = j
#             windows = []
#             if length <= self.window_size:
#                 windows.append(words)
#             else:
#                 for j in range(length - self.window_size + 1):
#                     window = words[j: j + self.window_size]
#                     windows.append(window)
#             word_pair_count = {}
#             for window in windows:
#                 for p in range(1, len(window)):
#                     for q in range(0, p):
#                         word_p = window[p]
#                         word_p_id = word_p # doc_word_id_map[word_p]
#                         word_q = window[q]
#                         word_q_id = word_q # doc_word_id_map[word_q]
#                         if word_p_id == word_q_id:
#                             continue
#                         word_pair_key = (word_p_id, word_q_id)
#                         # word co-occurrences as weights
#                         if word_pair_key in word_pair_count:
#                             word_pair_count[word_pair_key] += 1.
#                         else:
#                             word_pair_count[word_pair_key] = 1.
#                         # bi-direction
#                         word_pair_key = (word_q_id, word_p_id)
#                         if word_pair_key in word_pair_count:
#                             word_pair_count[word_pair_key] += 1.
#                         else:
#                             word_pair_count[word_pair_key] = 1.
#             row = []
#             col = []
#             weight = []
#             features = []

#             for key in word_pair_count:
#                 p = key[0]
#                 q = key[1]
#                 row.append(doc_word_id_map[p]) # p
#                 col.append(doc_word_id_map[q]) # q
#                 weight.append(word_pair_count[key])
#             adj = sp.csr_matrix((weight, (row, col)), shape=(len(doc_node), len(doc_node)))
#             batch_adj.append(preprocess_adj(adj.todense(),args.doc_len))
#         return torch.FloatTensor(np.array(batch_adj)).to(device)
# GD = GraphDataset(3)

# with open(doc_dict_path) as f:
#     for line in tqdm(f.readlines()):
#         l = line.strip().split('\t')
#         try:
#            doc_dict[l[0]] = [int(x) for x in l[1].split()]
#         except:
#            doc_dict[l[0]] = [138]

# for idx in range(32):
#     qrls, pos_docs, neg_docs = data_generator.sample()
#     unique_words = words_lookup(pos_docs)
#     pos_doc_list = []
#     for doc in pos_docs:
#         pos_doc_list.append(doc_dict[str(doc)])
#     adj_mat = GD.adjacent(unique_words,pos_doc_list)
#     print(adj_mat.shape)
#     break


# df = pd.read_csv("/media/sda2/Share/saran/GRMM/data/msmarco-passage/qrels/AAAI_project_test.tsv",sep='\t',header=None)
# qid_list = np.unique(df[0])
# x = np.zeros(54)
# for i,qid in enumerate(qid_list):
#    x[i] = len(df[df[0]==qid])
# print(x.max())

df = pd.read_csv("test.run")
print(df)