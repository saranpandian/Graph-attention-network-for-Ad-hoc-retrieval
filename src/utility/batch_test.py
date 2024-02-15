from utility.parser import parse_args
import scipy.sparse as sp
from utility.load_data import *
import subprocess
import torch
from tqdm import tqdm

args = parse_args()
device = torch.device("cuda:%d"%args.gpu_id)
model_name='GAT'
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
_, doc_dict_rev = data_generator.doc_dict, data_generator.doc_dict_rev
doc_dict = data_generator.doc_word_list
qrl_dict, qrl_dict_rev = data_generator.qrl_dict, data_generator.qrl_dict_rev
qrelf = args.data_path + args.dataset + '/qrels.train.tsv'
# doc_dict = {}
# doc_dict_path = "../data/msmarco-passage/map.documents.txt"
# with open(doc_dict_path) as f:
#     for line in tqdm(f.readlines()):
#         l = line.strip().split('\t')
#         try:
#            doc_dict[l[0]] = [int(x) for x in l[1].split()]
#         except:
#            doc_dict[l[0]] = [138]

def normalized_adj_bi(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_adj(adj,max_length, model='GRMM'):
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
    if model=='GAT':
       return np.array(list(adj)), mask
    else:
       return np.array(list(adj))

def pad_sequences(items, maxlen, value=data_generator.n_words):
    result = []
    for item in items:
        if len(item) < maxlen:
            item = item + [value] * (maxlen - len(item))
        if len(item) > maxlen:
            item = item[:maxlen]
        result.append(item)
    return result

def words_lookup(docs):
    if args.model == 'GRMM':
        return [data_generator.doc_unqiue_word_list[i] for i in docs]
    else:
        return [data_generator.doc_word_list[i] for i in docs]


class GraphDataset():

    def __init__(self,window_size,model='GRMM'):
        self.window_size = window_size
        self.model=model
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
            if self.model=='GRMM':
               batch_adj.append(preprocess_adj(adj.todense(),args.doc_len))
            else:
               adj, mask = preprocess_adj(adj.todense(),args.doc_len, self.model)
               batch_adj.append(adj)
               batch_mask.append(mask)
        if self.model=='GRMM':
           return np.array(batch_adj)
        else:
            return np.array(batch_adj), np.array(batch_mask)

GD = GraphDataset(3, model_name)

def test(model, qrls_to_test, model_name='GRMM'):
    test_qrls = qrls_to_test
    n_test_qrls = len(test_qrls)
    test_set = data_generator.test_set

    rate_batches = np.zeros(shape=(n_test_qrls, 368))
    count = 0

    for qrl_id in tqdm(test_qrls):
        doc_ids = test_set[qrl_id]

        docs_words = pad_sequences(words_lookup(doc_ids), maxlen=args.doc_len, value=138)

        qrls_words = [data_generator.qrl_word_list[qrl_id]]
        qrls_words = pad_sequences(qrls_words, maxlen=args.qrl_len, value=138)
        qrls_words = np.tile(qrls_words, [1, len(doc_ids)])
        qrls_words = np.reshape(qrls_words, [-1, args.qrl_len])
        qrls_words = torch.tensor(qrls_words).long().to(device)
        docs_words = torch.tensor(docs_words).long().to(device)
        unique_words = words_lookup(doc_ids)
        doc_list = []
        for doc in doc_ids:
                doc_list.append(doc_dict[doc])
        if model_name=='GRMM':
           adj_mat = GD.adjacent(unique_words, doc_list)
        else:
            adj_mat, mask_mat = GD.adjacent(unique_words, doc_list)
        with torch.no_grad():
            model.eval()
            if model_name=='GRMM':
               rate_batch = model(adj_mat, qrls_words, docs_words, doc_ids , test=True)
            else:
                rate_batch = model(adj_mat, mask_mat, qrls_words, docs_words, doc_ids , test=True)
            rate_batch = rate_batch.cpu()
        rate_batches[count, :rate_batch.shape[1]] = rate_batch
        count += 1

    runf = './test1.run'
    rerank_run = {}

    for i_qrl in range(len(test_qrls)):
        qn = test_qrls[i_qrl]
        for i_doc in range(len(test_set[qn])):
            dn = test_set[qn][i_doc]
            qid = qrl_dict_rev[qn]
            did = doc_dict_rev[dn]
            if dn not in test_set[qn]:
                continue
            rerank_run.setdefault(qid, {})[did] = rate_batches[i_qrl, i_doc]

    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run \n')            

    # trec_eval_f = 'bin/trec_eval'
    # output = subprocess.check_output([trec_eval_f, '-m', 'ndcg_cut.20', '-m', 'P.20', qrelf, runf]).decode().rstrip()

    # assert count == n_test_qrls
    # return eval(output.split()[2]), eval(output.split()[-1])
