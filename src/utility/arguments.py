import time
import random
import os
from sklearn import metrics
# from metrics import softmax_cross_entropy,accuracy
from argparse import ArgumentParser
# from utils import *
# from models import GNN
import numpy as np
import torch
import torch.nn as nn
# Set random seed
seed = 123

def set_seed(seed):
    random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(seed)
# parameters
parser = ArgumentParser()
parser.add_argument('-f')
parser.add_argument('--logger_name', help='Logger string.', default='TextING-pytorch')
parser.add_argument('--checkpoint_dir', help='Checkpoint string', default='checkpoints')
parser.add_argument('--dataset', help='Dataset string.', default='R8')
parser.add_argument('--model', help='Model string.', default='gnn')
parser.add_argument('--learning_rate', help='Initial learning rate.', default=0.0005)
parser.add_argument('--epochs', help='Number of epochs to train.', default=10)
parser.add_argument("--hidden_size_1", type=int, default=200, help="Size of first GCN hidden weights")
parser.add_argument("--hidden_size_2", type=int, default=100, help="Size of second GCN hidden weights")
parser.add_argument("--hidden_size_3", type=int, default=50, help="Size of second GCN hidden weights")
parser.add_argument("--hidden_dim", type=int, default=25, help="Size of second GCN hidden weights")
parser.add_argument('--batch_size', help='Size of batches per epoch.', default=4096) # 4096
parser.add_argument('--input_dim', help='Dimension of input.', default=300)
parser.add_argument("--num_classes", type=int, default=52, help="Number of prediction classes")
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=16, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# parser.add_argument('--hidden_dim', help='Number of units in hidden layer.', default=96)
parser.add_argument('--steps', help='Number of graph layers.', default=2)
parser.add_argument('--dropout1', help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--weight_decay', help='Weight for L2 loss on embedding matrix.', default=0)
parser.add_argument('--early_stopping', help='Tolerance for early stopping (# of epochs).', default=-1)
parser.add_argument('--max_degree', help='Maximum Chebyshev polynomial degree.', default=3) # not used
args_gat = parser.parse_args()
args_gat.device = 'gpu'