import os
import pickle
import sys
import time

#import networkx as nx
import psutil

import CreatIndex
#from fourSimiliarCompute.LR import load_weights_from_file
#from fourSimiliarCompute.getSimiliarMatrix import getSimiliarMatrix
#from foursimiliarget import foursimiliar, gettotalSimiliar
from fourSimiliarCompute.foursimiliarget import foursimiliar
# from myLogger import MyLogger
from page_ranks import IncrementalPersonalizedPageRank3


os.chdir('/home/lijiajun/infogether/')

# # 获取当前文件的目录路径
# current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# log_dir = os.path.join(current_dir, "log")

# os.makedirs(log_dir, exist_ok=True)
# # 拼接相对路径和日志文件名
# log_file_path = os.path.join(log_dir, "join.log")
# logger = MyLogger(log_file_path)


def pprMtrix(graph,num_walks,reset_pro,docnum,storepath):
    pprmatrixfile = os.path.join(storepath, "pprmatrix_n%s_r%s.pickle" % (num_walks, reset_pro))
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
    pprmatrixfile = os.path.join(storepath, "pprmatrix_n%s_r%s.pickle" % (num_walks, reset_pro))
    with open(pprmatrixfile, 'wb') as file:
        pickle.dump(page_ranks3, file)
    print("ppr matrix 构建完成")
    return page_ranks3


if __name__ == '__main__':


    # -------------------------------------创建索引-----------------------------------------

    # split 里面要对query数据进行查询，因此也需要把query 数据加入到索引的创建中，query 数据的路径为：/data_ssd/webtable/small_query/query

    filepath = "/data_ssd/webtable/large/split_1"
    queryfilepath = "/data_ssd/webtable/small_query/query"

    datalist = [filepath, queryfilepath]

    # 索引保存路径
    indexstorepath = "/data/lijiajun/infogather/webtables/index/"
    os.makedirs(indexstorepath,exist_ok=True)

    stime = time.time()
    try:
        CreatIndex.CreatDataIndex(datalist,indexstorepath)
    except Exception as e:
        print(e)
    etime = time.time()
    print(f"索引计算成功，计算索引时间为：{etime -stime}")
    

    # --------------------------------------计算四种相似度-------------------------------------
    doc_numberpkl = os.path.join(indexstorepath,"doc_number.pkl")
    with open(doc_numberpkl,"rb") as f:
        doc_number = pickle.load(f)
    print(f"doc number :{doc_number}")
    stime = time.time()
    foursimiliar(indexstorepath,doc_number)
    etime = time.time()
    print(f"comupute four  similiar end {etime - stime}")

    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024))


    # ## -----------------------------------计算相似度矩阵--------------------------------------
    # weigth = load_weights_from_file("/data/lijiajun/splitsmallindex")
    # weigths = [float(i) for i in weigth]
    # stime = time.time()
    # getSimiliarMatrix(indexstorepath,weigths)
    # etime = time.time()
    # print(f"similisr matrix compute {etime - stime}")


    # ## ---------------------------------构建 PPR 图------------------------------------
    # doc_number = 14861793
    # graph = nx.Graph()
    # start_time = time.time()
    # with open(os.path.join(indexstorepath, "similiarmatix.pkl"), 'rb') as f:
    #     similiar_matrix = pickle.load(f)

    # graph.add_nodes_from([i for i in range(doc_number)])
    # for i in similiar_matrix:
    #     for j in similiar_matrix[i].keys():
    #         # Get edges
    #         graph.add_edge(i, j)
    # end_time = time.time()
    # print("Graph construction successful: %s" % (end_time - start_time))
    

    # ##------------------------------计算PPR -----------------------------
    # Compute PPR
    ppr_vector = pprMtrix(graph, 300, 0.3, doc_number,indexstorepath)

    # ## queryonline3
    # # scan groundtruth  get indicate
    indicator(150, index, ppr_vector)





































