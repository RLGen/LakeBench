"""

这个文件是用来统计 union 在opendata 上进行查询的时候，占用的内存大小和平均每次查询占用的时间
用来得到论文中的指标

"""

import collections
import csv
import itertools
import os
import pickle
import time
from anyio import sleep
import networkx as nx
import numpy as np
import pandas as pd
import psutil

import CreatIndex
import util
from myLogger import MyLogger
from quaryOnline import QuaryOnline

from collections import defaultdict
import os
import pickle
import pandas as pd

import networkx as nx
import random
from numpy import cumsum, array
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
    return total_ppr


def indicator_union(ki,uninon_path = "/data_ssd/webtable/large/small_query",indexstorepath = "/data/lijiajun/infogather/webtables/index/"):

    

    
    inversedfile = os.path.join(indexstorepath,"ppr_matrix.pkl")
    with open(inversedfile, 'rb') as file:
        pprlist = pickle.load(file)
    

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

    docColumnsfile =  os.path.join(indexstorepath,"docColumns.pkl")
    with open(docColumnsfile, 'rb') as file:
        docColumns = pickle.load(file)

    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024))

    ## groundTrue dic
    
    querytableset = []
    for file in os.listdir(uninon_path):
        querytableset.append(file)

    final_union_dict = {}
    
    totaltime = 0

    stime = time.time()
    for querytable in tqdm(querytableset):
        # from querytable to get all column of this querytable
        #columns = index.docColumns[querytable]
        
        querytable_one = os.path.join(uninon_path,querytable)
        df = pd.read_csv(querytable_one,encoding= 'ISO-8859-1',low_memory=False,lineterminator='\n')
        columns = df.columns
        allreleventtable = set()
        allcolumns_ppr  = collections.defaultdict(dict)
        for column in columns:
            setresult=  querytable_columns(querytable_one,column,inversed,KIV,KIA,docNo,pprlist)
            allcolumns_ppr[column] = setresult
            table = set(map(lambda x: docNo.get(x,"").split("#")[0],setresult.keys()))
            allreleventtable = allreleventtable | table

        
        finaldic = {}
        for releventtable in allreleventtable:
            allcolumns = docColumns[releventtable]
            # 创建二分图
            G = nx.Graph()
            # 添加左侧节点（代理）
            G.add_nodes_from(list(columns),bipartite=0)
            # 添加右侧节点（接收者）
            G.add_nodes_from(list(allcolumns),bipartite=1)
            # 添加边和权重
            for leftnode in columns:
                for rightnode in allcolumns:
                    if rightnode in allcolumns_ppr[leftnode]:
                        G.add_edge(leftnode, rightnode, weight=allcolumns_ppr[leftnode][rightnode])

            # 使用最大权匹配算法
            matching = nx.max_weight_matching(G, weight='weight', maxcardinality=True)
            max_weight = 0
            tableColumsMatchList = []
            for left, right in matching:
                if not isinstance(right, int):
                    left,right = right,left
                weight = G[left][right]['weight']
                matchColumn = docNo[right].split("#")[1]
                tableColumsMatchList.append("%s,%s,%s"%(releventtable,left,matchColumn))
                max_weight += weight
            key = "%s#%s"%(releventtable,max_weight)
            finaldic.update({key:tableColumsMatchList})
        # 对 finaldic 按照权重进行排序，如果权重相同，对应value值中list长度最小排在前面
        sorteddic = sorted(finaldic.items(),key=lambda item:( - float(item[0].split("#")[1]),len(item[1])))
        # get top k relevent tables
        sorteddic = sorteddic[:ki]
        res_final = []
        for count, (ele,value) in enumerate(sorteddic):
            res_final.append(ele.split('#')[0])
        final_union_dict[querytable] = res_final
    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024/1024))

    etime = time.time()
    print(len(querytableset))
    print("timeconsumer ",etime - stime)
    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024))


    res_final_path = os.path.join(indexstorepath,"unin_res.pkl")
    with open(res_final_path,"wb") as f:
        pickle.dump(final_union_dict,f)
    
if __name__ == '__main__':

    ## queryonline
    # scan groundtruth  get indicate
    uninon_path = "/data_ssd/opendata/small/query"
    indexstorepath = "/data/lijiajun/infogather/opendata/index/"

    indicator_union(60,uninon_path,indexstorepath)














