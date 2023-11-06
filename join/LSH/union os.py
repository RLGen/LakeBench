from new import bootstrap_index_sets
import pickle
import os
import time, argparse, sys, json
import pdb
import numpy as np
import scipy.stats
import collections
import random
import pandas as pd
from pympler import asizeof
import farmhash
import multiprocessing 
import threading
import queue
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct
import csv
from collections import OrderedDict
import heapq

from datasketch import MinHashLSHEnsemble, MinHash

open_list = {1:"SG", 2:"UK", 3:"USA", 4:"CAN"}

id_now = 1
part_pattern = 1
sample_ratio = 1
data_partition = 1
if part_pattern != 3:
    data_partition = 1
# storage_path = "/data2/csy/web_large"

split_num = 40
split_num = min((multiprocessing.cpu_count(), split_num))
# query_path = "/data_ssd/webtable/small_unfake"
# index_path = "DataFilter/csvdata"
# index_path = "/data_ssd/webtable/large/split_"+str(id_now)
# index_path = "/data_ssd/opendata/small/datasets_"+open_list[id_now]
# index_path = "/data2/opendata/large/"+open_list[id_now]
# index_path = "/data_ssd/webtable/small_query/query"
index_paths = ["/data_ssd/opendata/small/datasets_"+open_list[id_now] for id_now in range(1,5)]
index_paths.append("/data_ssd/opendata/small/query")

# index_list = [os.path.join(storage_path, i) for i in ["index.pickle-1"]]
dict_cache = os.path.join("/data2/csy/dic", "distinct_dic_os.pickle")
# dict_cache_in = os.path.join(storage_path, "distinct_dic_w1.pickle")




def pre_multi_process(set_path, file_list, q, id):
    # print("{} start".format(id))
    datas = {}
    for i, sets_file in enumerate(file_list):
        try:
            df = pd.read_csv(sets_file, dtype='str', lineterminator='\n').dropna()
        except pd.errors.ParserError as e:
            print(sets_file)
        columns = df.columns.tolist()
        data = df.values.T.tolist()
        for column, vals in zip(columns, data):
            # 需要对value去重
            key = sets_file + "." + column
            datas[key] = list(set(vals))
        if id==0:
            sys.stdout.write("\rId 0 Process Read and pre {}/{} files".format(i+1, len(file_list)))
    if id==0:
        sys.stdout.write("\n")
    q.put((datas,))

def split_list(lst, num_parts):
    avg = len(lst) // num_parts
    remainder = len(lst) % num_parts

    result = []
    start = 0
    for i in range(num_parts):
        if i < remainder:
            end = start + avg + 1
        else:
            end = start + avg
        result.append(lst[start:end])
        start = end

    return result
    

if __name__ == "__main__":
    # 存储相关
    storage_config = {"type": "dict"}
    
    level = {
        "thresholds": [0.8],
        "num_parts": [8],
        "num_perms": [256],
        "m": 4
    }

    # index_data_cache_name = "/data2/csy/web_large/index.pickle-1"
    # # index_data_cache_name = "/data2/csy/open_small/index.pickle-2"
    
    # save_name = index_data_cache_name+"qq"
    # with open(index_data_cache_name, "rb") as d:
    #     index_data = pickle.load(d)
    # print("Using cached indexed sets {}".format(index_data_cache_name))

    # index_path = "/data_ssd/webtable/small_query/query"
    # indexed_tables = []
    # for filename in os.listdir(index_path):
    #     if filename.endswith(".csv") and "groundtruth" not in filename:
    #         indexed_tables.append(filename)
    # indexed_tables = [indexed_tables]
    # index_data_1 = bootstrap_index_sets(
    #             index_path, indexed_tables[0], num_perm=level["num_perms"][0],
    #             threshold=level["thresholds"][0], num_part=level["num_parts"][0], m=level["m"],
    #             storage_config=storage_config)
    # index_data[0].union(index_data_1[0])
    # with open(save_name, "wb") as d:
    #     pickle.dump(index_data, d)




    
    # 读入候选集目录
    indexed_tables = []

    for index_path in index_paths:
        for filename in os.listdir(index_path):
            if filename.endswith(".csv") and "groundtruth" not in filename:
                indexed_tables.append(os.path.join(index_path, filename))
    sets_files = indexed_tables

    sub_sets_files = split_list(sets_files, split_num)
    process_list = []
    # pocessing
    startr = time.perf_counter()
    pre_q = multiprocessing.Queue()
    distinct_dic = {}
    for i in range(split_num):
        process = multiprocessing.Process(target=pre_multi_process, args=(index_path, sub_sets_files[i], pre_q, i))
        process.daemon = True
        process_list.append(process)
        process.start()
    for i in range(split_num):
        res = pre_q.get()
        distinct_dic.update(res[0])
        sys.stdout.write("\rRead and size {}/{} sets".format(i+1,split_num))
    sys.stdout.write("\n")
    # pdb.set_trace()
    for i, process in enumerate(process_list):
        process.join()
    # print("build index time = {}, size = {} bytes".format(time.perf_counter() - startr, asizeof.asizeof(distinct_dic)))
    # with open(dict_cache_in, "rb") as d:
    #     data_dic = pickle.load(d)
    # distinct_dic.update(data_dic)
    with open(dict_cache, "wb") as d:
        pickle.dump(distinct_dic, d)

    
    
