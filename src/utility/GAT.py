import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import re
from torch.utils.data import DataLoader,Dataset
from GAT_model import *
from arguments import *

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from Early_stopping import *
patience = 4
early_stopping = EarlyStopping(patience=patience, verbose=True)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = list(set(nltk.corpus.stopwords.words("english")))

import torch
import torchtext.datasets
train_data,test_data = torchtext.datasets.DBpedia(root='.data', split=('train', 'test'))

final = []
count = 0
for x in train_data:
  temp=[]
  temp.append(x[0])
  temp.append(x[1])
  final.append(temp)

df = pd.DataFrame(final,columns=['labels','documents'])
df['labels'] = df['labels']-1
# df['labels'] = df['labels'].map({'pos':1,'neg':0})
print(df)
word_embeddings = {}
print("create glove embeddings")
with open("/home/irlab/Desktop/Graph_self_attention/"+'glove.6B.' + str(300) + 'd.txt', 'r') as f:
    for line in tqdm(f.readlines()):
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float,data[1:]))

def preprocess_adj(adj,max_length):
    # abj: [ns1xns2, ns2xns2....]
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # max_length = max([a.shape[0] for a in adj]) # a: node_size_a x node_size_a
    # mask: example_num x max_length x 1
    mask = np.zeros((max_length, 1)) # mask for padding #
    adj = adj+np.eye(len(adj))
    # for i in tqdm(range(adj.shape[0])):
    adj_normalized = normalize_adj(adj) # no self-loop
    pad = max_length - adj_normalized.shape[0] # padding for each epoch
    adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
    mask[:adj.shape[0],:] = 1
    adj = adj_normalized

    return np.array(list(adj)), mask

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1)) # D
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return np.matmul(adj,d_mat_inv_sqrt.transpose().dot(d_mat_inv_sqrt))

def preprocess_features(features,max_length):
    # features: [ns1x300, ns2x300, ...]
    """Row-normalize feature matrix and convert to tuple representation"""
    # max_length = max([len(f) for f in features])
    feature = np.array(features)
    pad = max_length - feature.shape[0] # padding for each epoch
    feature = np.pad(feature, ((0,pad),(0,0)), mode='constant')
    return np.array(list(feature))

class extract():
  def removePunctAndNumbers(self,xmlfile):
    res = re.sub(r'[^\w\s]', '', str(xmlfile) )
    res = re.sub("\d+", "", res)
    return res

f=extract()
class GraphDataset():
    def remove_pattern(self,input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(repr(i), '', input_txt)
        return input_txt
    def remove_stop_words(self,x):
        clean = []
        for i in x:
          if i not in stopwords:
            clean.append(i)
        return clean
    def preprocessing(self,doc_content_list,word_embeddings):
        unique_words = set()
        for doc_words in tqdm(list(doc_content_list['documents'])):
          doc_words = self.remove_pattern(doc_words,"@[\w]*").replace('\n',' ')
          doc_words = doc_words.replace("[^a-zA-Z]", " ")
          doc_words = ' '.join([w for w in doc_words.split() if len(w)>2])
          doc_words = f.removePunctAndNumbers(doc_words)
          doc_words = doc_words.lower().split()
          doc_words = self.remove_stop_words(doc_words)
          unique_words.update(doc_words)
        # for doc in list(doc_content_list['documents']):

        self.unique_words = list(unique_words)
        self.doc_word_embeddings = {}
        self.word_id_map = {}
        count = 0
        for word in self.unique_words:
          self.word_id_map[word] = count
          count+=1
          try:
            self.doc_word_embeddings[word] = word_embeddings[word]
          except:
            self.doc_word_embeddings[word] = np.random.uniform(-0.01, 0.01, 300)

    def __init__(self,MAX_TRUNC_LEN,truncate,window_size,doc_content_list,word_embeddings):
        self.MAX_TRUNC_LEN = MAX_TRUNC_LEN
        self.truncate = truncate
        self.window_size = window_size
        self.doc_content_list = doc_content_list
        self.word_id_map = None
        self.unique_words = None
        self.shuffle_doc_name_list = None
        self.label_list = None
        self.doc_content_list = doc_content_list
        self.preprocessing(doc_content_list,word_embeddings)

    def __getitem__(self,index):

        x_adj = []
        x_feature = []
        y = []
        doc_len_list = []
        vocab_set = set()

        doc_label1,doc_words = self.doc_content_list.iloc[index]
        doc_words = self.remove_pattern(doc_words,"@[\w]*")
        doc_words = doc_words.replace("[^a-zA-Z]", " ")
        doc_words = ' '.join([w for w in doc_words.split() if len(w)>2])
        doc_words = f.removePunctAndNumbers(doc_words)
        doc_words = doc_words.lower().split()
        doc_words = self.remove_stop_words(doc_words)
        if self.truncate:
            doc_words = doc_words[:self.MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= self.window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - self.window_size + 1):
                window = doc_words[j: j + self.window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = self.word_id_map[word_p] # doc_word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = self.word_id_map[word_q] # doc_word_id_map[word_q]
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
            row.append(doc_word_id_map[self.unique_words[p]]) # p
            col.append(doc_word_id_map[self.unique_words[q]]) # q
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(self.doc_word_embeddings[k])

        adj = np.array(adj.todense())
        x_feature = np.array(features)

    # one-hot labels
    # for i in range(start, end):
        # doc_meta = self.shuffle_doc_name_list[index]
        # temp = doc_meta.split('\t')
        # label = temp[2]
        # one_hot = [0 for l in range(len(self.label_list))]
        # label_index = self.label_list.index(label)
        # one_hot[label_index] = 1
        # y.append(one_hot)
        # y = np.array(y)
        # vocab_set = list(vocab_set)
        train_adj, train_mask = preprocess_adj(adj,self.MAX_TRUNC_LEN)
        train_feature = preprocess_features(x_feature,self.MAX_TRUNC_LEN)
        return train_adj, train_mask, train_feature, doc_label1

    def __len__(self):
        return len(self.doc_content_list)

MAX_TRUNC_LEN = 150
window_size=3
truncate=True
weighted_graph = False
train, validate = np.split(df.sample(frac=1, random_state=42),
                       [int(.9*len(df))])


mindDataset = GraphDataset(MAX_TRUNC_LEN,truncate,window_size,train,word_embeddings)

train_loader = torch.utils.data.DataLoader(mindDataset, batch_size=16,shuffle=True)

mindDataset = GraphDataset(MAX_TRUNC_LEN,truncate,window_size,validate,word_embeddings)

val_loader = torch.utils.data.DataLoader(mindDataset, batch_size=16,shuffle=True)

final = []
for x in test_data:
  temp = []
  temp.append(x[0])
  temp.append(x[1])
  final.append(temp)
df_test = pd.DataFrame(final,columns=['labels','documents'])
df_test['labels'] = df_test['labels'].map({'pos':1,'neg':0})

mindDataset = GraphDataset(MAX_TRUNC_LEN,truncate,window_size,df_test,word_embeddings)

test_loader = torch.utils.data.DataLoader(mindDataset, batch_size=16,shuffle=True)
del word_embeddings


def softmax_cross_entropy(loss_fn, preds, labels):
    loss = loss_fn(preds,labels)
    return loss


def accuracy(preds, labels):
    """Accuracy with masking."""
    return torch.sum(torch.argmax(preds,1) == labels) / len(preds)

#del word_embeddings
import tqdm
from sklearn.metrics import classification_report,confusion_matrix
from tqdm import tqdm
if args.model == 'gnn':
    model_func = GNN
import torch

model = model_func(args=args,
                   input_dim=args.input_dim,
                   output_dim=14,
                   hidden_dim=args.hidden_dim,
                   gru_step = args.steps,
                   dropout_p=args.dropout)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000,6000], gamma=0.77)
loss_fn = nn.CrossEntropyLoss()
loss_plot = []
acc_plot = []
for epoch in range(args.epochs):
    t = time.time()
    # Training step
    model.train()
    train_loss, train_acc, val_loss,val_acc = 0, 0, 0, 0
    for train_adj,train_mask,train_feature,train_y in tqdm(train_loader):
        outputs,_= model(train_feature.type(torch.cuda.FloatTensor), train_adj.type(torch.cuda.FloatTensor), train_mask.cuda()) # embeddings not used
        loss = softmax_cross_entropy(loss_fn, outputs, train_y.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(outputs, train_y.cuda())
        # scheduler.step()
        train_loss += loss.item()
        train_acc += acc.item()
    print('train loss: ',train_loss/len(train_loader))
    loss_plot.append(train_loss/len(train_loader))
    print('train accuracy: ',train_acc/len(train_loader))
    acc_plot.append(train_acc/len(train_loader))
    model.eval()
    pred = []
    truth = []
    t = time.time()
    for test_adj,test_mask,test_feature,test_y in tqdm(val_loader):
            # print(train_feature.shape)

            # print(train_adj.shape)
            # print(train_mask.shape)

        outputs,_= model(test_feature.type(torch.cuda.FloatTensor), test_adj.type(torch.cuda.FloatTensor), test_mask.cuda()) # embeddings not used
        acc = accuracy(outputs, test_y.cuda())
        val_acc+=acc.item()
        vali_loss = softmax_cross_entropy(loss_fn, outputs, test_y.cuda())
        val_loss += vali_loss.item()
    print('val loss: ',val_loss/len(val_loader))
    print('val acc:',val_acc/len(val_loader))
    loss_val = val_loss/len(val_loader)
    early_stopping(loss_val, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

test_loader = torch.utils.data.DataLoader(mindDataset, batch_size=16,shuffle=True)
test_acc,test_loss = 0,0
for test_adj,test_mask,test_feature,test_y in tqdm(test_loader):
    outputs,_= model(test_feature.type(torch.cuda.FloatTensor), test_adj.type(torch.cuda.FloatTensor), test_mask.cuda()) # embeddings not used
    acc = accuracy(outputs, test_y.cuda())
    test_acc+=acc.item()
    tes_loss = softmax_cross_entropy(loss_fn, outputs, test_y.cuda())
    test_loss += tes_loss.item()
print('test loss: ',test_loss/len(test_loader))
print('test acc:',test_acc/len(test_loader))