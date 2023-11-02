import os
import pickle
import time

import CreatIndex
from fourSimiliarCompute.foursimiliarget import foursimiliar
from page_ranks import IncrementalPersonalizedPageRank3
import networkx as nx


def creatInfogatherOffineIndex(datalist,indexstorepath = "/data/lijiajun/infogather/opendata/index/",
                               columnValue_rate=0.333,columnName_rate=0.3333,columnWith_rate=0.3333,
                               similar_thres= 0.7,values_maxlen_inverse = 200,dataset_large_or_small = "large",
                               num_walks = 300,reset_pro = 0.3):

    os.makedirs(indexstorepath,exist_ok=True)
    stime = time.time()
    try:
        CreatIndex.CreatDataIndex(datalist,indexstorepath)
    except Exception as e:
        print(e)
    etime = time.time()
    print(f"索引计算成功，计算索引时间为：{etime -stime}")
    

    # --------------------------------------comput foursimiliarity  , get similiarity matrix-------------------------------------
    doc_numberpkl = os.path.join(indexstorepath,"doc_number.pkl")
    with open(doc_numberpkl,"rb") as f:
        doc_number = pickle.load(f)
    foursimiliar(indexstorepath,doc_number,columnValue_rate,columnName_rate,columnWith_rate,similar_thres,values_maxlen_inverse)


    # ## -------------------------------------------graph construct -------------------------------------------------------------------
    filesimiliar = ""
    filesimiliar= os.path.join(indexstorepath,"similiarmatix.pkl")
    
    with open(filesimiliar, 'rb') as f:
        similiar_matrix = pickle.load(f)
    print("similiarmatix load ")

    docnumfile = os.path.join(indexstorepath,"doc_number.pkl")
    with open(docnumfile, 'rb') as f:
        docnum = pickle.load(f)
    print("doc_number load ")

    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(docnum)])
    for i in similiar_matrix:
        for j in similiar_matrix[i].keys():
            # Get edges
            graph.add_edge(i, j,weight=similiar_matrix[i][j])
    # store graph
    graphfile= os.path.join(indexstorepath,"graph_similiar.pkl")
    nx.write_gpickle(graph, graphfile)
    print("graph store success")

    # ##------------------------------compute PPR have two ways if dataset is small directly compute，else Calculate while storing  -----------------------------
    if dataset_large_or_small== "small":
        pr = IncrementalPersonalizedPageRank3(graph, num_walks, reset_pro,docnum)
        pr.initial_random_walks()
        page_ranks3 = pr.compute_personalized_page_ranks()
        fileppr = os.path.join(indexstorepath,"ppr_matrix.pkl")
        with open(fileppr ,"wb") as f:
            pickle.dump(page_ranks3,f)
        print("all the offline operations have done")
        # 离线处理完毕
    else:
        print("large dataset online store index")


        







































