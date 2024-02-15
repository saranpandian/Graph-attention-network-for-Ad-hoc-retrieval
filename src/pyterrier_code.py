# import transformers
# print(transformers.__version__)
import pyterrier as pt
import os
from unidecode import unidecode
os.environ["JAVA_HOME"] = "/home/irlab/java/jdk-11.0.17"
import pyterrier as pt
# from pyterrier import createDFIndex
from tqdm import tqdm
import numpy as np
# if not pt.started():
pt.init()
# from pyterrier_colbert.indexing import ColBERTIndexer
import unicodedata
# from pyterrier.measures import *

import pandas as pd
import inspect


import ir_datasets
import re
# dataset = ir_datasets.load('msmarco-passage')
df_test = pd.read_csv("test1.run",sep=' ',header=None)

df_test.columns = ['qid','Q0','docno','label','score','run','None']
# print(df_test.sort_values(by='score',ascending=False))
dataset_test = ir_datasets.load('msmarco-passage/trec-dl-2020')
doc_df_test = []
for doc in tqdm(dataset_test.queries_iter()):
    doc_df_test.append([doc.query_id,doc.text])
doc_df_test = pd.DataFrame(doc_df_test)
doc_df_test.columns = ['qid','query']

doc_df_qrels = []
for doc in tqdm(dataset_test.qrels_iter()):
       doc_df_qrels.append([doc.query_id,doc.doc_id, doc.relevance])
doc_df_qrels = pd.DataFrame(doc_df_qrels)
doc_df_qrels.columns = ['qid','docno','label']
doc_df_qrels.to_csv("qrels.csv")

from pyterrier.measures import *
print(pt.Experiment(
    [df_test[['qid','docno','score']]],
    doc_df_test,
    doc_df_qrels,
    eval_metrics=[RR(rel=2), nDCG@10, nDCG@100, AP(rel=2)],
    names=["GGRU"]
))