import collections
import csv
import itertools
import os
import pickle
import time
import networkx as nx
import numpy as np
import pandas as pd

import CreatIndex
import util
from page_ranks import IncrementalPersonalizedPageRank3
from quaryOnline import QuaryOnline




def get_groundTrue(file_path):
    """

    Args:
        file_path ():

    Returns:
        dict  {querytable:union tables and colums}

    """
    result = {}
    with open(file_path, 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key = row[0]
            if key == "query_table":
                continue
            value = row[1]
            if key in result:
                result[key].add(value)
            else:
                result[key] = {value}
    return result


def getTable0neColumnsUniqueValue(file_path,column):
    """

    Args:
        file_path ():groundTruth file path

    Returns:
         a list  all the table names in groundtruth
    """
    df = pd.read_csv(file_path)
    unique_values = df[column].unique()
    return set(unique_values)




def query_table_colums(querytable, querycolumn, index, ppr_vector, k):
    filepath = "D:\dataset\csv_benchmark"
    queryfile = os.path.join(filepath, querytable)
    q = QuaryOnline(queryfile, querycolumn, index, ppr_vector)
    q.getRelationtablesSore()
    result = q.getTransformatTableInfo(index.docNo)
    if not k:
        return result[querycolumn]
    return result[querycolumn][:k]

#file_ppr_list


def query_table_colums2(querytable, querycolumn, index, ppr_vector):
    filepath = "D:\dataset\csv_benchmark"
    queryfile = os.path.join(filepath, querytable)
    q = QuaryOnline(queryfile, querycolumn, index, ppr_vector)
    q.getRelationtablesSore()
    return q.file_ppr_list[querycolumn]


def pprMtrix(graph, num_walks, reset_pro, docnum):
    pprmatrixfile = "D:\ljj\data\pprmatrix_n%s_r%s.pickle" % (num_walks, reset_pro)
    if os.path.exists(pprmatrixfile):
        with open(pprmatrixfile, 'rb') as file:
            ppr_vector = pickle.load(file)
        print("no need to compute ppr")
        return ppr_vector
    start_time = time.time()
    pr = IncrementalPersonalizedPageRank3(graph, num_walks, reset_pro, docnum)
    pr.initial_random_walks()
    page_ranks3 = pr.compute_personalized_page_ranks()
    end_time = time.time()
    consumption_time = end_time - start_time
    print("Graph PPR calculation completed, computation time: %s" % consumption_time)
    pprmatrixfile = "D:\ljj\data\pprmatrix_n%s_r%s.pickle" % (num_walks, reset_pro)
    with open(pprmatrixfile, 'wb') as file:
        pickle.dump(page_ranks3, file)
    print("ppr matrix 构建完成")
    with open(pprmatrixfile, 'rb') as file:
        ppr_vector = pickle.load(file)
    return ppr_vector


def indicator_union(ki, index, ppr_vector):
    groundtrue = "D:\dataset\indicate\csv_groundtruth\\att_groundtruth.csv"
    ## groundTrue dic
    querytabledic = get_groundTrue(groundtrue)
    querytableset = querytabledic.keys()
    presicion_total = [0.0]*ki
    recall_total = [0.0]*ki
    totaltime = 0
    for querytable in querytableset:
        # from querytable to get all column of this querytable
        #columns = index.docColumns[querytable]
        stime = time.time()
        df = pd.read_csv(os.path.join("D:\dataset\csv_benchmark",querytable))
        columns = df.columns
        allreleventtable = set()
        allcolumns_ppr  = collections.defaultdict(dict)
        for column in columns:
            setresult = query_table_colums2(querytable, column, index, ppr_vector)
            allcolumns_ppr[column] = setresult
            table = set(map(lambda x: index.docNo.get(x,"").split("#")[0],setresult.keys()))
            allreleventtable = allreleventtable | table
        releventtablesore = collections.defaultdict(float)
        finaldic = {}
        for releventtable in allreleventtable:
            allcolumns = index.docColumns[releventtable]
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
                matchColumn = index.docNo[right].split("#")[1]
                tableColumsMatchList.append("%s,%s,%s"%(releventtable,left,matchColumn))
                max_weight += weight
            key = "%s#%s"%(releventtable,max_weight)
            finaldic.update({key:tableColumsMatchList})
        # 对 finaldic 按照权重进行排序，如果权重相同，对应value值中list长度最小排在前面
        sorteddic = sorted(finaldic.items(),key=lambda item:( - float(item[0].split("#")[1]),len(item[1])))
        etime = time.time()
        totaltime += etime - stime
        # get top k relevent tables
        sorteddic = sorteddic[:ki]
        recall = [0.0]*ki
        persicion = [0.0]*ki
        matchTrue = querytabledic[querytable]
        for count, (key,value) in enumerate(sorteddic):
            topktablename = key.split("#")[0]
            if topktablename in matchTrue:
                persicion[count] = 1
                recall[count] = 1/ len(matchTrue)
        persicion = [value/(ind + 1)  for ind,value in  enumerate(itertools.accumulate(persicion))]
        recall = list(itertools.accumulate(recall))
        print("query one time:%s"%(etime - stime))
        presicion_total = [x + y for x,y in zip(presicion_total,persicion)]
        recall_total = [x+y for x,y in zip(recall_total,recall)]
    tablenum = len(querytableset)
    presicion_total = [x / tablenum for x in presicion_total]
    recall_total = [y / tablenum for y in recall_total]

    totaltime = totaltime/tablenum
    print("30 percison:%s recall:%s" % (presicion_total[19], recall_total[19]))
    print("30 ave percison:%s ava recall:%s" % (sum(presicion_total[:20]) / 20, sum(recall_total[:20]) / 20))
    print("60 percison:%s recall:%s" % (presicion_total[59], recall_total[59]))
    print("60 ave percison:%s ava recall:%s" % (sum(presicion_total[:60]) / 60, sum(recall_total[:60]) / 60))
    print("150 percison:%s recall:%s" % (presicion_total[-1], recall_total[-1]))
    print("150 ave percison:%s ava recall:%s" % (sum(presicion_total) / ki, sum(recall_total) / ki))
    print("ave query time %s"%totaltime)

if __name__ == '__main__':
    # Create index

    filepath = "D:\dataset\csv_benchmark"
    # filepath = "D:\data_ljj"
    start_time = time.time()
    index = CreatIndex.CreatDataIndex(filepath)
    index.computeIndex()
    print("index length :%s" % index.doc_number)
    end_time = time.time()
    print("Index creation time: %s" % (end_time - start_time))
    with open("D:\ljj\data\my_index.pkl", "wb") as file:
        pickle.dump(index, file)

    # Compute similar matrix
    """
    similiar = MachineLearnSample(index)
    similiar.getlabled()
    """
    # Load graph
    graph = nx.Graph()
    start_time = time.time()
    with open("D:\ljj\data\similiarmatrix.pickle", 'rb') as f:
        similiar_matrix = pickle.load(f)

    graph.add_nodes_from([i for i in range(index.doc_number)])
    for tup in similiar_matrix:
        # Get edges
        edg1, edg2, edg3 = tup
        graph.add_edge(edg1, edg2)
    end_time = time.time()
    print("Graph construction successful: %s" % (end_time - start_time))
    print(index.docNo[1981])
    # Compute PPR
    ppr_vector = pprMtrix(graph, 300, 0.3, index.doc_number)

    ## queryonline
    # scan groundtruth  get indicate
    indicator_union(150, index, ppr_vector)














