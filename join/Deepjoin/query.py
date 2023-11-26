import pickle
import mlflow
import argparse
import time
import numpy as np
from hnsw_search import HNSWSearcher
# import tqdm.auto
import csv
import sys
import os
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="cl", choices=['sherlock', 'starmie', 'cl', 'tapex'])
    parser.add_argument("--benchmark", type=str, default='test')
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--single_column", dest="single_column", action="store_true", default=False)
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument("--scal", type=float, default=1.0)
    parser.add_argument("--mlflow_tag", type=str, default=None)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.7)

    hp = parser.parse_args()
    
    # mlflow logging
    for variable in ["encoder", "benchmark", "single_column", "run_id", "K", "scal"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    encoder = hp.encoder
    singleCol = False
    dataFolder = hp.benchmark
    K = hp.K
    threshold = hp.threshold
    N = hp.N

    # Set augmentation operators, sampling methods, K, and threshold values according to the benchmark
    if 'santos' in dataFolder or dataFolder == 'opendata':
        sampAug = "drop_cell_alphaHead"

    elif dataFolder == 'opendata' or dataFolder == 'test':
        sampAug = "drop_cell_alphaHead"
    singSampAug = "drop_col,sample_row_head"

    table_id = hp.run_id
    table_path = "/data/final_result/starmie/"+dataFolder+"/"+dataFolder+"_small_with_query.pkl"
    query_path = "/data/final_result/starmie/"+dataFolder+"/"+dataFolder+"_small_query.pkl"
    index_path = "/data/final_result/starmie/"+dataFolder+"/hnsw_opendata_small_"+str(table_id)+"_"+str(hp.scal)+".bin"

    # Call HNSWSearcher from hnsw_search.py
    searcher = HNSWSearcher(table_path, index_path, hp.scal)
    print(f"table_path: {table_path}")
    queries = pickle.load(open(query_path, "rb"))

    start_time = time.time()
    returnedResults = {}
    avgNumResults = []
    query_times = []

    dic = {}
    for qu in queries:
        str_q = qu[0].split('__')[0]
        if str_q not in dic:
            dic[str_q] = []
            # dic[str_q].append(qu)
        # else:
            # continue
        dic[str_q].append(qu)


path_output = '/data/final_result/starmie/'+dataFolder+'/result/small/hnsw_' + str(K) + '_' + str(N) + '_' + str(threshold) + '.csv'
if os.path.exists(path_output):
     os.remove(path_output)

for q in tqdm(queries):
    query_start_time = time.time()
    res, scoreLength = searcher.topk(encoder, q, K, N=N, threshold=threshold) #N=10,
    returnedResults[q[0]] = [r[2] for r in res]
    avgNumResults.append(scoreLength)
    query_times.append(time.time() - query_start_time)

    with open(path_output, 'a', encoding='utf-8', newline='') as file_writer:
        for i in range(0,len(res)):
            out_data = []
            out_data.append(q[0])
            out_data.append(res[i][2].split('/')[-1])
            for j in res[i][1]:
                out_data = []
                out_data.append(q[0])
                out_data.append(res[i][2].split('/')[-1])
                with open(q[0]+'.csv', 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    first_row = next(csvreader)
                    out_data.append(first_row[j[0]])
                if 'query' in res[i][2]:
                    path = res[i][2] + '.csv'
                    with open(path, 'r') as csvfile:
                        csvreader = csv.reader(csvfile)
                        first_row = next(csvreader)
                        out_data.append(first_row[j[1]])
                else:
                    path =res[i][2]+'.csv'
                    with open(path, 'r') as csvfile:                            
                        csvreader = csv.reader(csvfile)
                        first_row = next(csvreader)
                        out_data.append(first_row[j[1]]) 
                
                out_data.append(j[2])
                w = csv.writer(file_writer, delimiter=',')
                w.writerow(out_data)


# for q in queries:
#     if len(returnedResults[q[0]]) < K:
#         print(returnedResults[q[0]])