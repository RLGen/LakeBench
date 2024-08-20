import collections
import json
import multiprocessing
import os
import pdb
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
from block_and_verify import Pair, block, verify
from gensim.models import KeyedVectors
from Hierarchical_Grid import build_hierarchical_grid
from inverted_index import invert_index
from Pivot_Metric import find_outlier_pivots, pivot_based_mappings
from pympler import asizeof
import heapq

open_list = {1:"SG", 2:"UK", 3:"USA", 4:"CAN"}
rand_seed = 128
np.random.seed(rand_seed)
is_mul = True
is_local = False
# 本地
# is_part为真则只取部分且不分区 由sample_ratio确定
# 1:取全部 2：按比例抽样部分 3.分组
if is_local:
    is_rand = 0

    k = 3
    level = n_layers= 4
    tao = 0.3*2
    T = 0.4
    n_dims = 300
    base = 2
    storage_path = "PEXESO"
    query_results = os.path.join(storage_path, "PEXESO_benchmark_query_results.csv")
    index_data_cache = os.path.join(storage_path, "{}.pickle".format("index"))

    part_pattern = 2
    data_partition = 2
    sample_ratio = 0.015
    if part_pattern != 3:
        data_partition = 1

    split_num = 10
    split_num = min((multiprocessing.cpu_count(), split_num))

    query_path = index_path = "small/csv_benchmark"
    groundtruth_path = "small/csv_groundtruth/att_groundtruth.csv"
else:
# 服务器
    is_rand = 2
    id_now = 1

    k = 3
    level = n_layers= 4
    tao = 0.3*2
    T = 0.4
    n_dims = 50
    base = 2
    storage_path = "PEXESO"
    # ap_name = "easy"
    # ap_name = "large_50_"+open_list[id_now]
    # index_path = ["/data2/opendata/large/"+open_list[id_now] for id_now in range(1,7)]
    # index_path = "/data_ssd/opendata/small/datasets_"+open_list[id_now]

    # ap_name = "test"
    # index_path = ["small/"+str(id_now) for id_now in range(1,3)]

    ap_name = "os"
    index_path = ["/data_ssd/opendata/small/datasets_"+open_list[id_now] for id_now in range(1,5)]
    index_path.append("/data_ssd/opendata/small/query")

    groundtruth_path = "/data2/opendata/large_query/ground_truth.csv"
    query_path = "/data_ssd/opendata/small/query"

    # groundtruth_path = "small/csv_groundtruth/att_groundtruth.csv"
    # query_path = "small/csv_benchmark"

    query_results = os.path.join(storage_path, "PEXESO_benchmark_query_results_"+ap_name+".csv")
    index_data_cache = os.path.join(storage_path, "{}.pickle".format("index_"+ap_name))
    index_data_dic_cache = os.path.join(storage_path, "{}.pickle".format("dic_"+ap_name))
    pivot_data_cache = os.path.join(storage_path, "{}.pickle".format("pivot_"+ap_name))

    part_pattern = 1
    data_partition = 2
    sample_ratio = 0.01
    if part_pattern != 3:
        data_partition = 1

    split_num = 60
    split_num = min((multiprocessing.cpu_count(), split_num))

    query_col_part = True
    query_col_path = "/data_ssd/opendata/small/small_join.csv"

    dict_cache = os.path.join("/data2/csy/dic", "distinct_dic_os.pickle")##


top_nums = [10, 20, 30, 40, 50, 60]
time_threshold = 30


if is_rand==1:
    start = time.perf_counter()
    embedding_file = 'model/wiki-news-300d-1M.vec'
    model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    print("load model time = {}".format(time.perf_counter() - start))
elif is_rand==0:
    model = None
else:
    start = time.perf_counter()
    dic_data_cache = "model/glove.pikle"
    with open(dic_data_cache, "rb") as d:
        model = pickle.load(d)
    print("load model time = {}".format(time.perf_counter() - start))


# 将向量单位化
def unit_vector(vec):
    # 计算向量的模长
    vec_norm = np.linalg.norm(vec)
    # 将向量转化为单位向量
    unit_vec = vec / vec_norm
    return unit_vec

# 输出该值域对应的所有embedding 并返回单词到id映射
def find_word2vec_of_all(values, model, word_to_id=None):
    values = [value.lower() for value in values if value!=" " and value!=""]
    value_set = set(values)
    embeddings_of_all = []

    for value in value_set:
        tokens = value.split()
        embeddings = []
        for token in tokens:
            if is_rand==0:
                embedding = np.random.randn(n_dims)
                embeddings.append(embedding)
            elif is_rand==1:
                if token in model.key_to_index:
                    embedding = model[token]
                    embeddings.append(embedding)
            else:
                if token in model.keys():
                    embedding = model[token]
                    embeddings.append(embedding)
        if len(embeddings) != 0:
            if word_to_id is not None:
                word_to_id[value] = len(embeddings_of_all)
            array = np.array(embeddings)
            average_vector = np.mean(array, axis=0)
            average_vector = unit_vector(average_vector)
            embeddings_of_all.append(average_vector)

    return embeddings_of_all

# 将csv文件列表中每个表格中的column转换为二元组(ids, key)形式
# @profile
def bootstrap_sets(sets_files):
    # 从csv中读取加载
    print("Creating sets...")
    # sets_files = sets_files
    sets = collections.deque([])
    # all_str = set()
    keys = []
    # for sets_file in sets_files:
    #     df = pd.read_csv(os.path.join(query_path, sets_file), dtype='str').dropna()
    #     data = df.values.T.tolist()
    #     for vals in data:
    #         # 需要对value去重
    #         all_str = all_str | set(vals)

    # word_to_id = {word: idx for idx, word in enumerate(all_str)}
    word_to_id = {}
    # fasttext生成向量
    print("Start embedding...")
    all_emb = []
    # find_word2vec_of_all(all_str, model, word_to_id)
    # word_list = list(word_to_id.keys())

    for i,sets_file in enumerate(sets_files):
        df = pd.read_csv(sets_file, dtype='str', lineterminator='\n').dropna()
        columns = df.columns.tolist()
        data = df.values.T.tolist()
        for column,vals in zip(columns,data):
            
            # 需要对value去重
            vals = list(set(vals))
            ids = []
            for value in vals:
                value = value.lower()
                if value!=" " and value!="":
                    if value in word_to_id.keys():
                        ids.append(word_to_id[value])
                        continue
                    tokens = value.split()
                    embeddings = []
                    for token in tokens:
                        if is_rand==0:
                            embedding = np.random.randn(n_dims)
                            embeddings.append(embedding)
                        elif is_rand==1:
                            if token in model.key_to_index:
                                embedding = model[token]
                                embeddings.append(embedding)
                        else:
                            if token in model.keys():
                                embedding = model[token]
                                embeddings.append(embedding)
                    if len(embeddings) != 0:
                        word_to_id[value] = len(all_emb)
                        ids.append(len(all_emb))
                        array = np.array(embeddings)
                        average_vector = np.mean(array, axis=0)
                        average_vector = unit_vector(average_vector)
                        all_emb.append(average_vector)
            # mask = np.isin(vals, word_list)
            # if True in mask:
            #     ids = np.vectorize(lambda x: word_to_id[x])(vals[mask])
                # 列对应的向量id
            sets.append(ids)
            # 列的键值
            keys.append(sets_file+"."+column)
        sys.stdout.write("\rRead {}/{} files".format(i+1, len(sets_files)))
    sys.stdout.write("\n")

    sets = list(sets)

    return (sets, keys, all_emb)

# 获取query column在groundtruth表格中的对应结果
def get_reference(df, column_name):
    if is_local:
        filtered_df = df[df['query_table'] + '.' + df['query_col_name'] == column_name]
        reference_df = filtered_df['candidate_table'] + '.' + filtered_df['candidate_col_name']
    else:
        filtered_df = df[df['query_table'] + '.' + df['query_column'] == column_name]
        reference_df = filtered_df['candidate_table'] + '.' + filtered_df['candidate_column']
    return reference_df.tolist()

def get_col_name(key):
    table_name = key.split(".csv.")[0] + ".csv"
    table_name = table_name.split("/")[-1]
    col_name = key.split(".csv.")[1]
    return table_name+"."+col_name

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
    od = collections.OrderedDict()
    if len(res)<top_num:
        top_num = len(res)
    start = time.perf_counter()
    for i, item in enumerate(res):
        # set_path = "/data_ssd/webtable/large/split_1"
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

    # 获取索引表名并处理列数据
    indexed_tables = []
    with open(dict_cache, "rb") as d:
        data_dic = pickle.load(d)
    if os.path.exists(index_data_dic_cache):
        with open(index_data_dic_cache, "rb") as d:
            index_data_dic = pickle.load(d)
        index_data = index_data_dic["index_data"]
        pivots = index_data_dic["pivots"]
        xys = index_data_dic["xys"]
        # 计算 轴上的分区边界
        x_min, x_max = np.floor(np.min(xys)), np.ceil(np.max(xys))
        # 首先划分支点，基于level
        lidu = base ** level
        a = (x_max - x_min) / lidu
        
        index_grid_struct = index_data_dic["grid"]
        I = index_data_dic["invert_index"]

    else:
        index_data_dic = {}
        if os.path.exists(index_data_cache):
            print("Using cached indexed sets {}".format(index_data_cache))
            with open(index_data_cache, "rb") as d:
                index_data = pickle.load(d)
        else:
            random.seed(rand_seed)
            for per_index_path in index_path:
                for filename in os.listdir(per_index_path):
                    if filename.endswith(".csv") and "groundtruth" not in filename:
                        if part_pattern == 2:
                            if random.random() > sample_ratio:
                                continue
                        indexed_tables.append(os.path.join(per_index_path,filename))

            start = time.perf_counter()
            index_data = bootstrap_sets(indexed_tables)
            print("(index)embedding data time = {}".format(time.perf_counter() - start))
            # with open(index_data_cache, "wb") as d:
            #     pickle.dump(index_data, d)
        index_data_dic["index_data"] = index_data
    
        # 处理pivot
        start = time.perf_counter()
        if os.path.exists(pivot_data_cache):
            print("Using cached pivot sets {}".format(pivot_data_cache))
            with open(pivot_data_cache, "rb") as d:
                pivots = pickle.load(d)
        else:
            pivots = find_outlier_pivots(index_data[2], k)
            with open(pivot_data_cache, "wb") as d:
                pickle.dump(pivots, d)
        print("(index)find_outlier_pivots time = {}".format(time.perf_counter() - start))
        index_data_dic["pivots"] = pivots

        # pivot映射
        points = index_data[2]
        start = time.perf_counter()
        xys = pivot_based_mappings(points, pivots)
        print("(index)pivot_based_mappings time = {}".format(time.perf_counter() - start))
        index_data_dic["xys"] = xys

        # 建层次网格索引
        start = time.perf_counter()
        # 计算 轴上的分区边界
        x_min, x_max = np.floor(np.min(xys)), np.ceil(np.max(xys))
        # 首先划分支点，基于level
        lidu = base ** level
        a = (x_max - x_min) / lidu
        index_grid_struct, id_to_grid = build_hierarchical_grid(k, xys, points, level, x_min, x_max, a)
        print("(index)build_hierarchical_grid time = {}".format(time.perf_counter() - start))
        index_data_dic["grid"] = index_grid_struct

        start = time.perf_counter()
        # 创建倒排索引
        I = invert_index()
        for i, ids in enumerate(index_data[0]):
            for id in ids:
                I.add(id_to_grid[id], i)
        print("(index)build_invert_index time = {}".format(time.perf_counter() - start))
        index_data_dic["invert_index"] = I

        print("index vectors size = {} bytes".format(asizeof.asizeof(index_data)))
        print("build index size = {} bytes".format(asizeof.asizeof(I)+asizeof.asizeof(index_grid_struct)))            
        with open(index_data_dic_cache, "wb") as d:
            pickle.dump(index_data_dic, d)

    print("{} columns and {} vectors".format(len(index_data[0]), len(index_data[2])))
    
    query_tables = []
    if is_local:
        # 随机取10张query表
        df = pd.read_csv(groundtruth_path, dtype='str').dropna()
        column_data = df["query_table"]
        unique_values = column_data.unique()
        np.random.seed(rand_seed)
        random_values = np.random.choice(unique_values, size=10, replace=False)
        query_tables = random_values
    else:
        if query_col_part:
            df = pd.read_csv(query_col_path, dtype='str').dropna()
            column_data = df["table_name"]
            unique_values = column_data.unique()
            query_tables = unique_values.tolist()
        else:
            for filename in os.listdir(query_path):
                if filename.endswith(".csv") and "groundtruth" not in filename:
                    query_tables.append(filename)


    num = 0
    times = []
    precisions = []
    recalls = []
    results = []
    query_keys = []
    match_results = {}
    for i in top_nums:
        match_results[i] = []
    groundtruth_df = pd.read_csv(groundtruth_path, dtype='str').dropna()
    if query_col_part:
        query_df = pd.read_csv(query_col_path, dtype='str').dropna()

    for sets_file in query_tables:
        
        df = pd.read_csv(os.path.join(query_path,sets_file), dtype='str', lineterminator='\n').dropna()
        columns = df.columns.tolist()
        data = df.values.T.tolist()
        if query_col_part:
            query_col_lines = query_df[query_df['table_name'] == sets_file]
            query_col_list = query_col_lines['col_name'].to_list()
        for column, vals in zip(columns, data):
            if query_col_part:
                if column not in query_col_list:
                    continue
            query_key = sets_file+"."+column
            qstart = time.perf_counter()
            num += 1
            # if num>1:##
            #     break
            # 需要对value去重
            vals = list(set(vals))
            # 列的键值
            query_embs = find_word2vec_of_all(vals, model)
            
            if len(query_embs)==0:
                continue
            query_keys.append(get_col_name(query_key))
            sys.stdout.write("\rquery {} columns".format(num))
            print(" Query column name:{},len:{}".format(column,len(vals)))
            start = time.perf_counter()
            query_vecs = pivot_based_mappings(query_embs, pivots)
            print("(query)pivot_based_mappings time = {}".format(time.perf_counter() - start))
            
            start = time.perf_counter()
            query_grid_struct, _ = build_hierarchical_grid(k, query_vecs, query_embs, level, x_min, x_max, a)
            print("(query)build_hierarchical_grid = {}".format(time.perf_counter() - start))

            Pairs = {}
            match_map = {}
            mismatch_map = {}
            for key in range(len(index_data[0])):
                match_map[key] = 0
                mismatch_map[key] = 0
            
            start = time.perf_counter()
            block(query_grid_struct.root, index_grid_struct.root, Pairs, tao)
            print("block time = {}".format(time.perf_counter() - start))

            res, over_time = verify(Pairs, I, tao, T*len(vals), index_data[2], query_embs, index_data[0], match_map, mismatch_map, time_threshold)
            
            if not over_time:
                times.append(time.perf_counter() - qstart)
            else:
                print("bad column:{}".format(query_key))
            
            res = [index_data[1][i] for i in res]
            results.append(res)
            for top_num in top_nums:
                ress = get_top(query_key, res, top_num)
                ress = [get_col_name(i) for i in ress]
                match_results[top_num].append((query_key, ress))
            # 记录精度召回
            reference = get_reference(groundtruth_df, query_key)
            precision, recall = get_precision_recall(res, reference)
            precisions.append(precision)
            recalls.append(recall)
            
    sys.stdout.write("\n")

    # 整理结果并保存
    if os.path.exists(query_results):
        df = pd.read_csv(query_results).dropna()
    else:
        rows = []
        # precisions, recalls, times, results
        
        Pe_res_path = "Pe_results_"+ap_name
        if os.path.exists(Pe_res_path):
            os.removedirs(Pe_res_path)
        os.mkdir(Pe_res_path)
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
                columns=["query_table", "candidate_table", "query_col_name", "candidate_col_name"])
            df_match.to_csv(os.path.join(Pe_res_path, "Pe-"+ap_name+"-top"+str(top_num)+"-match.csv"), index=False)

        for probe_time, precision, recall, result, query_key in zip(\
                times, precisions, recalls, results, query_keys):
            rows.append((query_key, probe_time, precision, recall, result))
        df = pd.DataFrame.from_records(rows,
            columns=["query_key", "query_time", "precision", "recall", "result"])
        df.to_csv(query_results, index=False)
    
    precision = df["precision"].mean()
    std = df["precision"].std()
    print("precision_mean = {}, std = {}".format(precision, std))
    recall = df["recall"].mean()
    std = df["recall"].std()
    print("recall_mean = {}, std = {}".format(recall, std))
    t = df["query_time"].mean()
    print("query time = {}".format(t))
