import ir_datasets
import numpy as np
import pandas as pd
# fold = 5
# dataset = ir_datasets.load("msmarco-passage/trec-dl-hard/fold{}".format(fold))
dataset = ir_datasets.load("msmarco-passage/trec-dl-2020")
final_list = []
for qrel in dataset.qrels_iter():
    if qrel.relevance>=2:
       final_list.append([qrel.query_id, qrel.doc_id, 1])
    else:
        final_list.append([qrel.query_id, qrel.doc_id, 0])
df_test = pd.DataFrame(final_list)
# # # dataset = ir_datasets.load("msmarco-passage/train")
# # # df = pd.read_csv("data/msmarco-passage/qrels.train.tsv",sep='\t',header=None)
# df_test.columns = ['query_id','doc_id','relevance']
df_test = df_test.merge(df,on = 'query_id')
df_test.astype(int).to_csv("AAAI_project_test.tsv",sep='\t',index=False,header=False)


# df1 = pd.read_csv("AAAI_Project.csv")
# df2 = pd.read_csv("AAAI_Project_dev.csv")
# df3 = pd.read_csv("AAAI_project_test.csv")
# df1[['query_id','doc_id','relevance']].astype(int).to_csv("AAAI_Project.tsv",sep='\t',index=False,header=False)
# df2[['query_id','doc_id','relevance']].astype(int).to_csv("AAAI_Project_dev.tsv",sep='\t',index=False,header=False)
# df3[['query_id','doc_id','relevance']].astype(int).to_csv("AAAI_project_test.tsv",sep='\t',index=False,header=False)
# print(pd.concat([df1,df2,df3])[['query_id','doc_id','relevance']].astype(int).to_csv("data/msmarco-passage/qrels.train.tsv",sep='\t',index=False,header=False))

# df = pd.read_csv("data/msmarco-passage/queries/queries.eval.tsv",sep='\t',header=None).astype(str)
# df.columns = ['query_id','query']

# df_test = df_test.merge(df,on = 'query_id')
# print(df_test)
# print(len(np.unique(df_test['query_id'])))


# df1 = pd.read_csv("data/msmarco-passage/queries/queries.dev.tsv",sep='\t',header=None)
# df2 = pd.read_csv("data/msmarco-passage/queries/queries.eval.tsv",sep='\t',header=None)
# df3 = pd.read_csv("data/msmarco-passage/queries/queries.train.tsv",sep='\t',header=None)


# pd.concat([df1,df2,df3]).to_csv("data/msmarco-passage/queries.tsv",sep='\t',index=False,header=False)