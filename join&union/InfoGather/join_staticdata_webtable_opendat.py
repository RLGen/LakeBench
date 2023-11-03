"""

This file is used to calculate the memory usage and average time spent on each query by the join when querying on opendata and webtable
Used to obtain the indicators in the paper
If you want to calculate the online memory size and average query time of the join on the webtable, you can restore the relevant annotations of the webtable instead
This is not on queryonline_ Opendata and queryonline_ The reason for directly calculating online memory usage and average query time on webtable is:
1. Due to the large amount of small data in webtable and opendata, it takes a long time to directly calculate PPR index references. Therefore, when querying, calculate the relevant indexes as needed
Then store the calculated index
2. The most accurate method is to load the stored index and then calculate the online memory usage size and average query time

"""


from collections import defaultdict
import os
import pickle
import time
import pandas as pd

import networkx as nx
import random
from numpy import cumsum, array
import psutil
from tqdm import tqdm




def querytable_columns(tablefile,column,inversed,KIV,KIA,docNo,pprdict):
    df = pd.read_csv(tablefile,encoding= 'ISO-8859-1',low_memory=False,lineterminator='\n')
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


#---------------------static  openda dataset when join get the size of memory and average time on line
filepath = "/data_ssd/opendata/small/small_join.csv"
querytablepath= "/data_ssd/opendata/small/query"
indexstorepath = "/data/lijiajun/infogather/opendata/index"

#---------------------static  webtable dataset  when join get the size of memory and average time on line
filepath = "/data_ssd/opendata/small/small_join.csv"
# webtable dataset
querytablepath= ["/data_ssd/webtable/large/split_1","/data_ssd/webtable/large/small_query"]

indexstorepath = "/data/lijiajun/infogather/webtables/index/"


fileppr = os.path.join(indexstorepath,"ppr_matrix.pkl")
with open(fileppr ,"rb") as f:
    pprdict = pickle.load(f)

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

print("graph load")


df = pd.read_csv(filepath)
resultdict = {}

total_rows = len(df)
stime = time.time()
# use tqdm 
for index, row in tqdm(df.iterrows(), total=total_rows):
    try:
        querytable = row[0]
        column = row[1]
        if querytable and column:
            queryfilepath = os.path.join(querytablepath,querytable)
            res_list = querytable_columns(queryfilepath,column,inversed,KIV,KIA,docNo,pprdict)
            key = querytable + str("###") + column
            resultdict[key] = res_list
    except Exception as e:
        print(row)
        print(e)
process = psutil.Process(os.getpid())
print("current memory：%s" % (process.memory_info().rss / 1024 / 1024/1024))
etime = time.time()

print("time consumer ",etime - stime)
print("ok")

