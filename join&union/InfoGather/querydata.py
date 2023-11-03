from collections import defaultdict
import os
import pickle
import pandas as pd

import networkx as nx
import random
from numpy import cumsum, array
from tqdm import tqdm

indexstorepath = "/data/lijiajun/infogather/webtables/index/"

inversedfile = os.path.join(indexstorepath,"inversed_index.pkl")
with open(inversedfile, 'rb') as file:
    inversed = pickle.load(file)

KIVfile =  os.path.join(indexstorepath,"KIV.pkl")
with open(KIVfile, 'rb') as file:
    KIV = pickle.load(file)
    
KIAfile =  os.path.join(indexstorepath,"KIA.pkl")
with open(KIAfile, 'rb') as file:
    KIA = pickle.load(file)

docNofile =  os.path.join(indexstorepath,"docNo.pkl")
with open(docNofile, 'rb') as file:
    docNo = pickle.load(file)

graphfile= os.path.join(indexstorepath,"graph_similiar.pkl")
#加载文件为图对象
graph = nx.read_gpickle(graphfile)

print("graph load")


from page_ranks import IncrementalPersonalizedPageRank1

def querytable_columns(tablefile,column,inversed,KIV,KIA,graph,docNo):
    df = pd.read_csv(tablefile)
    columnvalues = df[column].unique()

    columnvalues_len = len(columnvalues)

    KIA_docs_set = set(KIA[column])

    KIV_docs_list = []
    for value in columnvalues:
        KIV_docs_list += inversed[value]

    KIV_docs_set = set(KIV_docs_list)

    KIV_KIA_set = KIA_docs_set & KIV_docs_set

    # compute weight
    weigth_dict = {}
    for doc in KIV_KIA_set:
        docvalue_set = set(KIV[doc])
        doclen = len(docvalue_set)
        overlapevalues = docvalue_set & set(columnvalues)
        weight = len(overlapevalues) / min(doclen,columnvalues_len)
        weigth_dict[doc] = weight

    # compute ppr
    pr = IncrementalPersonalizedPageRank1(graph, 300, 0.3,list(KIV_KIA_set))
    pr.initial_random_walks()
    pprdict = pr.compute_personalized_page_ranks()

    # get all the weigted dict
    total_ppr_list = []
    for doc  in KIV_KIA_set:
        doc_weigth = weigth_dict[doc]
        doc_dict_weigth = {key:value*doc_weigth for key,value in pprdict[doc].items()}
        total_ppr_list.append(doc_dict_weigth)
    
    # compute the  weighted dict
    total_ppr = defaultdict(float)
    for ppr_ele in total_ppr_list:
        for key,value in ppr_ele.items():
            total_ppr[key] += value
    
    sorted_dict = dict(sorted(total_ppr.items(), key=lambda x: x[1],reverse=True))

    return_doc_list = list(sorted_dict.keys())[:30]

    return_name = []
    for docnumber in return_doc_list:
        return_name.append(docNo[docnumber])

    return return_name
# queryfilepath = "/data_ssd/webtable/large/split_1/csvData10000001.csv"
# columnname= "Sample Date"
# res = querytable_columns(queryfilepath,columnname,inversed,KIV,KIA,graph,docNo)


filepath = "/data_ssd/webtable/large/small_join.csv"
indexstorepath = "/data/lijiajun/infogather/webtables/index/"

df = pd.read_csv(filepath)
resultdict = {}

total_rows = len(df)

# 使用 tqdm 添加进度条
for index, row in tqdm(df.iterrows(), total=total_rows):
    try:
        querytable = row[0]
        column = row[1]
        if querytable and column:
            if "__" not in querytable:
                queryfilepath = os.path.join("/data_ssd/webtable/large/split_1",querytable)
            else:
                queryfilepath = os.path.join("/data_ssd/webtable/large/small_query",querytable)
            res_list = querytable_columns(queryfilepath,column,inversed,KIV,KIA,graph,docNo)
            key = querytable + str("###") + column
            resultdict[key] = res_list
    except Exception as e:
        print(e)

finalres_file = os.path.join(indexstorepath,"final_res_dic.pkl")
with open(finalres_file,"wb") as f:
    pickle.dump(resultdict,f)

obj = IncrementalPersonalizedPageRank1.ppr_vectors
fileppr = os.path.join(indexstorepath,"ppr_matrix.pkl")
with open(fileppr ,"wb") as f:
    pickle.dump(obj,f)

print("运行完成")



