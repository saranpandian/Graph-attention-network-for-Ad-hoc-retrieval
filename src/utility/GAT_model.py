import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .arguments import *

def glorot(shape):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = nn.Parameter(torch.nn.init.uniform_(tensor=torch.empty(shape), a=-init_range, b=init_range))
    return initial

def xavier(shape):
    initial = nn.Parameter(torch.nn.init.xavier_uniform_(tensor=torch.empty(shape), gain=1))
    return initial


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.W = glorot([in_features, out_features])
        self.a = glorot([2*out_features, 1])
        self.bias = nn.Parameter(torch.zeros(out_features))
        # self.a_bias1 = nn.Parameter(torch.zeros(1))
        # self.a_bias2 = nn.Parameter(torch.zeros(1))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # print(h.type(torch.cuda.FloatTensor).shape)
        # print(self.W.shape)
        Wh = torch.matmul(h.type(torch.cuda.FloatTensor), self.W) + self.bias # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2,1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid,hidden_dim, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, hidden_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hidden_dim * nheads, nfeat, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj,mask):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(mask*self.out_att(x, adj))
        return x

class GraphLayer(nn.Module):
    """Graph layer."""
    def __init__(self, args,
                      input_dim,
                      output_dim,
                      act=nn.Tanh(),
                      dropout_p = 0.,
                      gru_step = 2):
        super(GraphLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.gru_step = gru_step
        self.gru_unit = GAT(input_dim, args.hidden,args.hidden_dim, args.dropout1, args.alpha, args.nb_heads)
        # self.dropout
        self.encode_weight = glorot([self.input_dim, self.input_dim])
        self.encode_bias = nn.Parameter(torch.zeros(self.input_dim))


    def forward(self, feature, support,mask):
        feature = self.dropout(feature)
        # encode inputs
        # print(feature.dtype)
        # print(self.encode_weight.dtype)
        encoded_feature = torch.matmul(feature, self.encode_weight) + self.encode_bias
        output = mask * self.act(encoded_feature)
        # convolve
        # for _ in range(self.gru_step):
        output = self.gru_unit(output,support,mask)
        return output

class ReadoutLayer(nn.Module):
    """Graph Readout Layer."""
    def __init__(self, args,
                 input_dim,
                 output_dim,
                 act=nn.ReLU(),
                 dropout_p=0.):
        super(ReadoutLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.att_weight = glorot([self.input_dim, 1])
        self.emb_weight = glorot([self.input_dim, self.input_dim])
        self.mlp_weight = glorot([self.input_dim, self.output_dim])
        self.att_bias = nn.Parameter(torch.zeros(1))
        self.emb_bias = nn.Parameter(torch.zeros(self.input_dim))
        self.mlp_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self,x,_,mask):  # _ not used
        # soft attention
        if args_gat.device == 'gpu':
           x = x.type(torch.cuda.FloatTensor)
           mask = mask.type(torch.cuda.FloatTensor)
        else:
           x = x.type(torch.FloatTensor)
           mask = mask.type(torch.FloatTensor)
        att = torch.sigmoid(torch.matmul(x, self.att_weight)+self.att_bias)
        emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)
        N = torch.sum(mask, dim=1)
        M = (mask - 1) * 1e9
        # graph summation
        g = mask * torch.bmm(att.transpose(2,1),emb)
        g = torch.sum(g, dim=1)/N + torch.max(g+M,dim=1)[0]
        g = self.dropout(g)
        # classification
        output = torch.matmul(g,self.mlp_weight)+self.mlp_bias
        return output


class GNN(nn.Module):
    def __init__(self, args, input_dim, output_dim, hidden_dim, gru_step, dropout_p):
        super(GNN,self).__init__()
        self.args = args
        print(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.gru_step = gru_step
        self.GraphLayer = GraphLayer(
            args = args,
            input_dim = self.input_dim,
            output_dim = self.hidden_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p,
            gru_step = self.gru_step
        )
        self.ReadoutLayer = ReadoutLayer(
            args=args,
            input_dim = self.hidden_dim,
            output_dim = self.output_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p
        )
        self.layers = [self.GraphLayer]


    def forward(self, feature, support, mask):
        activations = [feature]

        for layer in self.layers:
            hidden = layer(activations[-1], support, mask)
            activations.append(hidden)
        embeddings = activations[-2]
        outputs = activations[-1]
        return outputs,embeddings