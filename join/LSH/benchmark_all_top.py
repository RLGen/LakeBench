import ast
import time, argparse, sys, json
import pdb
import numpy as np
import scipy.stats
import collections
import random
import os
import pickle
import pandas as pd
from pympler import asizeof
import farmhash
import multiprocessing 
import threading
import queue
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct
import csv
from collections import OrderedDict, defaultdict
import heapq

from datasketch import MinHashLSHEnsemble, MinHash

open_list = {1:"SG", 2:"UK", 3:"USA", 4:"CAN"}
rand_seed = 41
is_spark = False
is_mul = True
is_local = False
is_query = False
is_top = True
has_groundtruth = True
top_num = 20
ap_str = "top20"
# 本地 
# is_part为真则只取部分且不分区 由sample_ratio确定
# 1:取全部 2：按比例抽样部分 3.分组
if is_local:
    id_now = 1
    part_pattern = 1
    sample_ratio = 1
    data_partition = 1
    if part_pattern != 3:
        data_partition = 1
    storage_path = "LSH Ensemble"

    split_num = 10 
    split_num = min((multiprocessing.cpu_count(), split_num))

    query_path = index_path = "small/csv_benchmark"
    groundtruth_path = "small/csv_groundtruth/att_groundtruth.csv"

    index_list = [os.path.join(storage_path, i) for i in ["index.pickle-1"]]
    dict_cache = os.path.join(storage_path, "distinct_dic.pickle")##
else:
# 服务器
    ap_name = "wl"
    id_now = 1
    part_pattern = 1
    sample_ratio = 1
    data_partition = 1
    if part_pattern != 3:
        data_partition = 1

    split_num = 40
    split_num = min((multiprocessing.cpu_count(), split_num))
    # query_path = "/data_ssd/webtable/small_unfake"
    query_path = "/data_ssd/webtable/large/large_query/"
    groundtruth_path = "/data2/opendata/large_query/ground_truth.csv"
    # index_path = "DataFilter/csvdata"

    storage_path = "/data2/csy/web_large"##
    index_path = ["/data_ssd/webtable/large/split_"+str(id_now) for id_now in range(1, 7)]
    index_path.append(query_path)
    index_list = [os.path.join(storage_path, i) for i in ["index.pickle-1","index.pickle-2"]]##
    dict_cache = os.path.join("/data2/csy/dic", "distinct_dic_wl.pickle")##

    # storage_path = "/data2/csy/open_small"##
    # index_path = "/data_ssd/opendata/small/datasets_"+open_list[id_now]
    # index_list = [os.path.join(storage_path, i) for i in ["index.pickle-1"]]##
    # dict_cache = os.path.join(storage_path, "distinct_dic_o1.pickle")##

    query_col_part = True
    query_col_path = "/data_ssd/webtable/large/large_join.csv"


data_dic = {}
query_tables = []
indexed_tables = []
top_nums = [5, 10, 15, 20, 25, 30]



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


def _hash_32(d):
    return farmhash.hash32(d)


# f1指标
def fscore(precision, recall):
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2.0 / (1.0 / precision + 1.0 / recall)


def average_fscore(founds, references):
    return np.mean([fscore(*get_precision_recall(found, reference))
                    for found, reference in zip(founds, references)])


# 计算精度和召回
def get_precision_recall(found, reference):
    reference = set(reference)
    intersect = sum(1 for i in found if i in reference)
    if len(found) == 0:
        precision = 0.0
    else:
        precision = float(intersect) / float(len(found))
    if len(reference) == 0:
        recall = 1.0
    else:
        recall = float(intersect) / float(len(reference))
    if len(found) == len(reference) == 0:
        precision = 1.0
        recall = 1.0
    return [precision, recall]


def _compute_containment(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = len(np.intersect1d(x, y, assume_unique=True))
    return float(intersection) / float(len(x))

def get_top(key, res, top_num):
    query_set = get_sets(query_path, key)
    od = OrderedDict()
    if len(res)<top_num:
        top_num = len(res)
    start = time.perf_counter()
    for i, item in enumerate(res):
        # set_path = "/data_ssd/webtable/large/split_1"
        # set_path = index_path
        # if "__" in item.split(".csv.")[0]:
        #     set_path = "/data_ssd/webtable/small_query/query"
        # key = "csv"+item.split("/csv")[-1]
        res_set = data_dic[item]
        od[item] = _compute_containment(query_set, res_set)
    
    # print("compute time = {}".format(time.perf_counter() - start))
    # start = time.perf_counter()
    result = list(heapq.nlargest(top_num, od.items(), key=lambda x: x[1]))
    # print("sort time = {}".format(time.perf_counter() - start))
    # pdb.set_trace()
    result = [key for key, _ in result]
        
    return result

def get_sets(path, key):
    table_name = key.split(".csv.")[0] + ".csv"
    col_name = key.split(".csv.")[1]
    df = pd.read_csv(os.path.join(path, table_name), dtype='str').dropna()
    data = df[col_name].tolist()
    return list(set(data))

if __name__ == "__main__":
    # 解析指令
    match_results = {}
    for i in top_nums:
        match_results[i] = []

    # 创建一个defaultdict，用于存储合并后的结果
    merged_data = defaultdict(set)

    # 遍历多个CSV文件
    file_paths = ["/data2/csy/web_large/lshensemble_benchmark_query_results_wl_23.csv",
                  "/data2/csy/web_large/lshensemble_benchmark_query_results_wl_1.csv",
                  "/data2/csy/web_large/lshensemble_benchmark_query_results_wl_6.csv",
                  "/data2/csy/web_large/lshensemble_benchmark_query_results_wl_45.csv"]  # 用实际的文件路径替换这些文件名
    csv.field_size_limit(1000000000)
    for file_path in file_paths:
        n = 0
        print("Start read {}".format(file_path))
        with open(file_path, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                n += 1
                key = row[0]  # 第一列为键值 
                # values = set(row[8].split(","))  # 第二列为集合，用逗号分隔并转换为集合
                values = ast.literal_eval(row[8])
                merged_data[key].update(values)  # 将集合添加到字典中的对应键值下
                sys.stdout.write("\rRead {}/{} lines".format(n, 6812))
        sys.stdout.write("\n")

    # 将defaultdict转换为普通的字典
    merged_dict = dict(merged_data)

    # 打印合并后的字典
    # print(merged_dict)
    
    with open(dict_cache, "rb") as d:
        data_dic = pickle.load(d)
    # print("data dic size = {} bytes".format(asizeof.asizeof(data_dic)))

    n = 0
    for (query_key, result) in merged_dict.items():
        n += 1
        result = list(result)
        sys.stdout.write("\rsort {}/{} lines".format(n, 6812))
        for top_num in top_nums:
            ress = get_top(query_key, result, top_num)
            match_results[top_num].append((query_key, ress))
    sys.stdout.write("\n")
        
    Lsh_res_path = os.path.join(storage_path, "Lsh_results_"+ap_name)
    if os.path.exists(Lsh_res_path):
        os.removedirs(Lsh_res_path)
    os.mkdir(Lsh_res_path)

    for top_num in top_nums:
        match_rows = []
        for row in match_results[top_num]:
            query_table = row[0].split(".csv.")[0] + ".csv"
            query_col_name = row[0].split(".csv.")[1]
            for candidate in row[1]:
                match_rows.append((query_table,
                                candidate.split(".csv.")[0] + ".csv",
                                query_col_name,
                                candidate.split(".csv.")[1]))
        df_match = pd.DataFrame.from_records(match_rows,
                                            columns=["query_table", "candidate_table", "query_col_name",
                                                    "candidate_col_name"])
        # 输出匹配 同att_groundtruth格式
        match_filename = "match_"+ap_name+"top_"+str(top_num)+".csv"
        df_match.to_csv(os.path.join(Lsh_res_path, match_filename), index=False)

