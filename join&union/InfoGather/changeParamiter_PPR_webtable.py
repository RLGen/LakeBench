import os
import pickle
import time

import networkx as nx
import numpy as np





def pprMtrix2(similiar_matrix,main_node,docnum):

    stime = time.time()
    graph = nx.Graph()
    # data=[(0, 1), (1, 2), (1, 3), (2, 4)]
    # for ele in data:
    #     x,y = ele[0],ele[1]
    #     graph.add_edge(x, y,weight=0.5)

    graph.add_nodes_from([i for i in range(docnum)])
    for i in similiar_matrix:
        for j in similiar_matrix[i].keys():
            # Get edges
            graph.add_edge(i, j,weight=similiar_matrix[i][j])


    etime = time.time()

    print("graph load",etime-stime)

    start_time = time.time()
    page_ranks_2 = nx.pagerank(graph, alpha=0.7, personalization={main_node: 1},
                               max_iter=100, weight='weight')
    end_time = time.time()
    print("Power Iteration Pageranks time: ", end_time -start_time)

    start_time = time.time()
    pr = IncrementalPersonalizedPageRank2(graph, 300, 0.3,docnum)
    pr.initial_random_walks()
    page_ranks3 = pr.compute_personalized_page_ranks()
    end_time = time.time()
    print("Mokar tero Pageranks time: ", end_time - start_time)

    print(page_ranks3)
    re = []
    for i in range(docnum):
        if i not in page_ranks3[0].keys():
            re.append(0.0)
        else:
            re.append(page_ranks3[0][i])
    page_ranks_array = np.array(re)
    #page_ranks_array = np.array(list(page_ranks_2_2.values()))
    page_ranks_2_array = np.array(list(page_ranks_2.values()))


    # 计算差异并进行归一化
    difference = np.linalg.norm(page_ranks_array - page_ranks_2_array) / np.linalg.norm(page_ranks_2_array)

    # 打印结果
    return difference




if __name__ == '__main__':

    indexstorepath = "/data/lijiajun/infogather/opendata/index"
    filesimiliar = ""
    if not filesimiliar:
        filesimiliar= os.path.join(indexstorepath,"similiarmatix.pkl")
    
    start_time = time.time()
    with open(filesimiliar, 'rb') as f:
        similiar_matrix = pickle.load(f)
    
    print("similiarmatix load ")
## print index all the numeber ---------------------------------------------------------
    docnumfile = os.path.join(indexstorepath,"doc_number.pkl")
    with open(docnumfile, 'rb') as f:
        docnum = pickle.load(f)
    print(docnum)
    
    print("doc_number load ")
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(docnum)])
    for i in similiar_matrix:
        for j in similiar_matrix[i].keys():
            # Get edges
            graph.add_edge(i, j,weight=similiar_matrix[i][j])
    print("grah success")
    # 存储图为文件
    graphfile= os.path.join(indexstorepath,"graph_similiar.pkl")
    nx.write_gpickle(graph, graphfile)
    print("graph store success")





#---------------------------------------------加载图 文件--------------------------------------
    # stime = time.time()
    # graphfile= os.path.join(indexstorepath,"graph_similiar.pkl")
    # #加载文件为图对象
    # loaded_graph = nx.read_gpickle(graphfile)
    # etime = time.time()
    # print("grapth loads",etime - stime)


    # num_walks = 300
    # reset_pro = 0.3
    # stime = time.time()
    # pr = IncrementalPersonalizedPageRank3(graph, num_walks, reset_pro,docnum)
    # pr.initial_random_walks()
    # page_ranks3 = pr.compute_personalized_page_ranks()
    # etime = time.time()
    # print("ppr 构建时间：",etime - stime)

    # pprmatrixfilename = "pprmatrix_n%s_r%s.pickle" % (num_walks, reset_pro)
    # pprmatrixfile = os.path.join(indexstorepath,pprmatrixfilename)
    # with open(pprmatrixfile, 'wb') as file:
    #     pickle.dump(page_ranks3, file)
    # print("ppr matrix 构建完成")






