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
from collections import OrderedDict
import heapq

from datasketch import MinHashLSHEnsemble, MinHash

open_list = {1:"SG", 2:"UK", 3:"USA", 4:"CAN"}
rand_seed = 41
is_spark = False
is_mul = True
is_local = False
is_query = True
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
    index_num_list = [4,5]
    res_name = ""
    for i in index_num_list:
        res_name += str(i)
    index_list = [os.path.join(storage_path, "index.pickle-"+str(i)) for i in index_num_list]##
    # index_list.append(os.path.join(storage_path, "index_wlq.pickle-1"))
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


def pre_multi_process(file_list, q, id):
    # print("{} start".format(id))
    sizes = []
    for i, sets_file in enumerate(file_list):
        try:
            df = pd.read_csv(sets_file, dtype='str', lineterminator='\n').dropna()
        except pd.errors.ParserError as e:
            print(sets_file)
        data = df.values.T.tolist()
        for vals in data:
            # 需要对value去重
            sizes.append(len(set(vals)))
        if id==0:
            sys.stdout.write("\rId 0 Process Read and pre {}/{} files".format(i+1, len(file_list)))
    if id==0:
        sys.stdout.write("\n")
    q.put((sizes,))
    # print("{} end,size={}".format(id,len(sizes)))

def minhash_multi_process(file_list, q, param_dic, id):
    # print("{} start".format(id))
    lsh = MinHashLSHEnsemble(threshold=param_dic['threshold'], num_perm=param_dic['num_perm'],
                             num_part=param_dic['num_part'], m=param_dic['m'], storage_config=param_dic['storage_config'])
    for i, (lower, upper) in enumerate(param_dic["partitions"]):
        lsh.lowers[i], lsh.uppers[i] = lower, upper
    for i, sets_file in enumerate(file_list):
        df = pd.read_csv(sets_file, dtype='str', lineterminator='\n').dropna()
        columns = df.columns.tolist()
        data = df.values.T.tolist()
        for column, vals in zip(columns, data):
            # 需要对value去重
            vals = list(set(vals))
            # 域的键值
            key = sets_file + "." + column
            # 生成minhash
            mh = MinHash(param_dic['num_perm'], hashfunc=_hash_32)
            for word in vals:
                mh.update(str(word))
            lsh.index((key, mh, len(vals)))
        if id==0:
            sys.stdout.write("\rId 0 Process Read and minhash {}/{} files".format(i+1, len(file_list)))
    if id==0:
        sys.stdout.write("\n")
    q.put(lsh)

# 将csv文件列表中每个表格中的column转换为三元组(minhashes, sets, keys)形式
# @profile
def bootstrap_sets(sets_files, num_perm):
    # 从csv中读取加载
    print("Creating sets...")
    print("Using num_perm = {}".format(num_perm))
    sizes = []
    keys = []
    # random.seed(rand_seed)
    minhashes = dict()
    ms = []
    if query_col_part:
        query_df = pd.read_csv(query_col_path, dtype='str').dropna()
    for sets_file in sets_files:
        df = pd.read_csv(os.path.join(query_path, sets_file), dtype='str').dropna()
        columns = df.columns.tolist()
        data = df.values.T.tolist()
        if query_col_part:
            query_col_lines = query_df[query_df["table_name"] == sets_file]
            query_col_list = query_col_lines['col_name'].to_list()
        # pdb.set_trace()
        for column, vals in zip(columns, data):
            if query_col_part:
                if column not in query_col_list:
                    continue
            # if random.random() > sample_ratio:
            #     continue
            # 需要对value去重
            vals = list(set(vals))
            # s = np.array(vals)
            # sets.append(s)
            # 域的键值
            keys.append(sets_file + "." + column)
            sizes.append(len(vals))
            # 生成minhash
            m = MinHash(num_perm, hashfunc=_hash_32)
            for word in vals:
                m.update(str(word))
            ms.append(m)

            sys.stdout.write("\rRead and minhashed {} sets".format(len(keys)))
    sys.stdout.write("\n")

    # [hash函数个数][各集合]
    minhashes[num_perm] = ms

    return (minhashes, sizes, keys)


# 将csv文件列表中每个表格中的column转换为三元组(minhashes, sets, keys)形式
# @profile
def bootstrap_index_sets(sets_files, num_perm, threshold, num_part, m, storage_config):
    # 从csv中读取加载
    print("Creating sets...")
    print("Using num_perm = {}".format(num_perm))

    # 处理完整路径
    # sets_files = [os.path.join(set_path, sets_file) for sets_file in sets_files]
    start = time.perf_counter()
    lsh = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm,
                             num_part=num_part, m=m, storage_config=storage_config)

    startr = time.perf_counter()

    # 读入候选集目录
    sub_sets_files = split_list(sets_files, split_num)
    process_list = []
    # pocessing
    pre_q = multiprocessing.Queue()
    sizes = []
    for i in range(split_num):
        process = multiprocessing.Process(target=pre_multi_process, args=(sub_sets_files[i], pre_q, i))
        process.daemon = True
        process_list.append(process)
        process.start()
    for i in range(split_num):
        res = pre_q.get()
        sizes.extend(res[0])
        sys.stdout.write("\rRead and size {}/{} sets".format(i+1,split_num))
    sys.stdout.write("\n")
    for i, process in enumerate(process_list):
        process.join()

    # threading
    # pre_q = queue.Queue()
    # sizes = []
    # for i in range(split_num):
    #     process = threading.Thread(target=pre_multi_process, args=(set_path, sub_sets_files[i], pre_q))
    #     process_list.append(process)
    #     process.start()
    # for i in range(split_num):
    #     res = pre_q.get()
    #     sizes.extend(res[0])
    #     for key in res[1]:
    #         name_to_id[key] = len(id_to_name)
    #         id_to_name.append(key)
    #     sys.stdout.write("\rRead and size {}/{} sets".format(i+1,split_num))
    # sys.stdout.write("\n")
    # for i, process in enumerate(process_list):
    #     process.join()
        
    print("size time = {}".format(time.perf_counter() - startr))
    print("size num = {}".format(len(sizes)))
    partitions = lsh.count_partition(sizes)

    entries_q = multiprocessing.Queue()
    param_dic = {'threshold':threshold, 'num_perm':num_perm, 'num_part':num_part, 'm':m, 'storage_config':storage_config, "partitions":partitions}
    for i in range(split_num):
        process = multiprocessing.Process(target=minhash_multi_process, args=(sub_sets_files[i], entries_q, param_dic, i))
        process.daemon = True
        process_list.append(process)
        process.start()
    for i in range(split_num):
        lsh.union(entries_q.get())
        sys.stdout.write("\rRead and minhash {}/{} sets".format(i+1,split_num))
    sys.stdout.write("\n")
    for i, process in enumerate(process_list):
        process.join()

    # print("build index time = {}".format(time.perf_counter() - start))
    print("build index time = {}, size = {} bytes".format(time.perf_counter() - start, asizeof.asizeof(lsh)))
    return (lsh,)


# 获取query column在groundtruth表格中的对应结果
def get_reference(df, column_name):
    # pdb.set_trace()
    if is_local:
        filtered_df = df[df['query_table'] + '.' + df['query_col_name'] == column_name]
        reference_df = filtered_df['candidate_table'] + '.' + filtered_df['candidate_col_name']
    else:
        filtered_df = df[df['query_table'] + '.' + df['query_column'] == column_name]
        reference_df = filtered_df['candidate_table'] + '.' + filtered_df['candidate_column']
    return reference_df.tolist()


# 对lshensemble进行测试 返回时间与查询结果
def benchmark_lshensemble(threshold, num_perm, num_part, m, storage_config,
                          lsh_list, query_data):
    print("Querying")
    if has_groundtruth:
        df = pd.read_csv(groundtruth_path, dtype='str').dropna()
    # lsh = index_data[0]
    # (lsh,) = index_data
    (minhashes, sizes, keys) = query_data
    probe_times = []
    precisions = []
    recalls = []
    results = []
    # match_results = {}
    # for i in top_nums:
    #     match_results[i] = []
    pre_list = ["/data_ssd/webtable/large/split_4/", "/data_ssd/webtable/large/split_5/"]
    for qsize, minhash, query_key in zip(sizes, minhashes[num_perm], keys):
        # 记录查询时间
        if qsize == 0:
            continue
        start = time.perf_counter()
        result = []
        # pdb.set_trace()
        for id, lsh in enumerate(lsh_list):
            now_res = list(lsh[0].query(minhash, qsize))
            now_res = [pre_list[id]+i for i in now_res]
            result.extend(now_res)
        # result = [id_to_name[res] for res in result]
        result = get_top(query_key, result, 30)
        # for top_num in top_nums:
        #     ress = get_top(query_key, result, top_num)
        #     match_results[top_num].append((query_key, ress))
        probe_times.append(time.perf_counter() - start)
        print(probe_times[-1])
        results.append(result)
        # 记录精度召回
        if has_groundtruth:
            reference = get_reference(df, query_key)
            precision, recall = get_precision_recall(result, reference)
        else:
            precision = recall = 0
        precisions.append(precision)
        recalls.append(recall)
        sys.stdout.write("\rQueried {} sets".format(len(probe_times)))
        # if len(recalls)>20:
        #     break
    sys.stdout.write("\n")
    return precisions, recalls, probe_times, results


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

levels = {
    "test": {
        "thresholds": [0.8],
        "num_parts": [8],
        "num_perms": [256],
        "m": 4,
    },
    "test2": {
        "thresholds": [0.8],
        "num_parts": [8],
        "num_perms": [256],
        "m": 2,
    },
    "lite": {
        "thresholds": [0.5, 0.75, 1.0],
        "num_parts": [8, 16],
        "num_perms": [32, 64],
        "m": 8,
    },
    "medium": {
        "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "num_parts": [8, 16, 32],
        "num_perms": [32, 128, 224],
        "m": 8,
    },
    "complete": {
        "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "num_parts": [8, 16, 32],
        "num_perms": [32, 64, 96, 128, 160, 192, 224, 256],
        "m": 8,
    },
}

if __name__ == "__main__":
    # 解析指令
    parser = argparse.ArgumentParser(
        description="Run LSH Ensemble benchmark using data sets obtained "
                    "from https://github.com/ekzhu/set-similarity-search-benchmarks.")
    # parser.add_argument("--indexed-sets", type=str, required=True,
    #         help="Input indexed set file (gzipped), each line is a set: "
    #         "<set_size> <1>,<2>,<3>..., where each <?> is an element.")
    # parser.add_argument("--query-sets", type=str, required=True,
    #         help="Input query set file (gzipped), each line is a set: "
    #         "<set_size> <1>,<2>,<3>..., where each <?> is an element.")
    parser.add_argument("--query-results", type=str,
                        default=os.path.join(storage_path, "lshensemble_benchmark_query_results_"+ap_name+"_"+res_name+".csv"))
    parser.add_argument("--ground-truth-results", type=str,
                        default="lshensemble_benchmark_ground_truth_results.csv")
    parser.add_argument("--indexed-sets-sample-ratio", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--level", type=str, choices=levels.keys(),
                        default="test")
    parser.add_argument("--use-redis", action="store_true")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--partID", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])

    level = levels[args.level]

    if args.threshold != 0:
        level["thresholds"][0] = args.threshold

    index_data, query_data = None, None
    index_data_cache = os.path.join(storage_path, "{}.pickle".format("index_"+ap_name))
    query_data_cache = os.path.join(storage_path, "{}.pickle".format("query_"+ap_name))
    
    print("multiprocessing split_num = {}".format(split_num))

    # 存储相关
    storage_config = {"type": "dict"}
    if args.use_redis:
        storage_config = {
            "type": "redis",
            "redis": {
                "host": args.redis_host,
                "port": args.redis_port,
            },
        }

    # 随机取100张query表
    if is_query and is_local:
        df = pd.read_csv(groundtruth_path, dtype='str').dropna()
        column_data = df["query_table"]
        unique_values = column_data.unique()
        np.random.seed(rand_seed)
        random_values = np.random.choice(unique_values, size=100, replace=False)
        query_tables = random_values.tolist()

    # 将querytables中所有表作为query表
    if is_query and not is_local:
        if query_col_part:
            df = pd.read_csv(query_col_path, dtype='str').dropna()
            column_data = df["table_name"]
            unique_values = column_data.unique()
            query_tables = unique_values.tolist()
            # query_tables = [os.path.join(query_path, i) for i in query_tables]
        else:
            for filename in os.listdir(query_path):
                if filename.endswith(".csv") and "groundtruth" not in filename:
                    query_tables.append(filename)

    # 查询数据缓存
    # if is_query:
    with open(dict_cache, "rb") as d:
        data_dic = pickle.load(d)
    # print("data dic size = {} bytes".format(asizeof.asizeof(data_dic)))
    if os.path.exists(query_data_cache):
        print("Using cached query sets {}".format(query_data_cache))
        with open(query_data_cache, "rb") as d:
            query_data = pickle.load(d)
    else:
        # print("Using query sets {}".format(query_tables))
        start = time.perf_counter()
        query_data = bootstrap_sets(query_tables, num_perm=level["num_perms"][0]) #只执行第一个设定的num_perms值
        print("minhash query_data time = {}".format(time.perf_counter() - start))
        with open(query_data_cache, "wb") as d:
            pickle.dump(query_data, d)

    # 将csvdata中所有表作为index表
    random.seed(rand_seed)
    if part_pattern == 3:
        for filename in os.listdir(index_path):
            if filename.endswith(".csv") and "groundtruth" not in filename:
                indexed_tables.append(filename)
        indexed_tables = split_list(indexed_tables, data_partition)
    else:
        for per_index_path in index_path:
            for filename in os.listdir(per_index_path):
                if filename.endswith(".csv") and "groundtruth" not in filename:
                    if part_pattern == 2:
                        if random.random() > sample_ratio:
                            continue
                    indexed_tables.append(os.path.join(per_index_path, filename))
        indexed_tables = [indexed_tables]

    lsh_list = []
    if not is_query:
        # 索引数据缓存
        if part_pattern == 3:
            part_id = id_now = args.partID
        else:
            part_id = 0
        
        index_data_cache_name = index_data_cache + "-" + str(id_now)
        if os.path.exists(index_data_cache_name):
            print("Using cached indexed sets {}".format(index_data_cache_name))
            # with open(index_data_cache_name, "rb") as d:
            #     index_data = pickle.load(d)
        else:
            # print("Using indexed sets {}".format(indexed_tables))
            start = time.perf_counter()
            index_data = bootstrap_index_sets(
                indexed_tables[part_id], num_perm=level["num_perms"][0],
                threshold=level["thresholds"][0], num_part=level["num_parts"][0], m=level["m"],
                storage_config=storage_config)
            print("minhash index_data time = {}".format(time.perf_counter() - start))
            with open(index_data_cache_name, "wb") as d:
                pickle.dump(index_data, d)



    if is_query:
        # 运行lshensamble，获取相关指标，整理结果并保存
        if os.path.exists(args.query_results):
            df = pd.read_csv(args.query_results).dropna()
        else:
            for path in index_list:
                with open(path, "rb") as d:
                    lsh_list.append(pickle.load(d))
                print("Using cached indexed sets {}".format(path))
            rows = []
            for threshold in level["thresholds"]:
                for num_part in level["num_parts"]:
                    for num_perm in level["num_perms"]:
                        print("Running LSH Ensemble benchmark "
                            "threshold = {}, num_part = {}, num_perm = {}".format(
                            threshold, num_part, num_perm))
                        precisions, recalls, probe_times, results = benchmark_lshensemble(
                            threshold, num_perm, num_part, level["m"], storage_config,
                            lsh_list, query_data)
                        
                        # Lsh_res_path = os.path.join(storage_path, "Lsh_results_"+ap_name)
                        # if os.path.exists(Lsh_res_path):
                        #     os.removedirs(Lsh_res_path)
                        # os.mkdir(Lsh_res_path)
                        # for top_num in top_nums:
                        #     match_rows = []
                        #     for row in match_results[top_num]:
                        #         query_table = row[0].split(".csv.")[0] + ".csv"
                        #         query_col_name = row[0].split(".csv.")[1]
                        #         for candidate in row[1]:
                        #             match_rows.append((query_table,
                        #                             candidate.split(".csv.")[0] + ".csv",
                        #                             query_col_name,
                        #                             candidate.split(".csv.")[1]))
                        #     df_match = pd.DataFrame.from_records(match_rows,
                        #                                         columns=["query_table", "candidate_table", "query_col_name",
                        #                                                 "candidate_col_name"])
                        #     # 输出匹配 同att_groundtruth格式
                        #     match_filename = str(threshold) + "-" + str(num_part) + "-" + str(num_perm) + "-" + "match_"+ap_name+"top_"+str(top_num)+".csv"
                        #     df_match.to_csv(os.path.join(Lsh_res_path, match_filename), index=False)

                        for probe_time, precision, recall, result, query_size, query_key in zip( \
                                probe_times, precisions, recalls, results, \
                                query_data[1], query_data[2]):
                            rows.append((query_key, query_size, threshold,
                                        num_part, num_perm, probe_time, precision, recall, result))
            df = pd.DataFrame.from_records(rows,
                                        columns=["query_key", "query_size", "threshold", "num_part",
                                                    "num_perm", "query_time", "precision", "recall", "result"])
            df.to_csv(args.query_results, index=False)

        # 按参数划分
        thresholds = sorted(list(set(df["threshold"])))
        num_perms = sorted(list(set(df["num_perm"])))
        num_parts = sorted(list(set(df["num_part"])))

        # 指标整理
        for i, num_perm in enumerate(num_perms):
            for j, num_part in enumerate(num_parts):
                for k, threshold in enumerate(thresholds):
                    print("Thresholds = {}, num_perms = {}, num_parts = {}".format(threshold, num_perm, num_part))
                    sub = df[(df["num_part"] == num_part) & (df["num_perm"] == num_perm) & (df["threshold"] == threshold)]
                    precision = sub["precision"].mean()
                    std = sub["precision"].std()
                    print("precision_mean = {}, std = {}".format(precision, std))
                    recall = sub["recall"].mean()
                    std = sub["recall"].std()
                    print("recall_mean = {}, std = {}".format(recall, std))
                    t = sub["query_time"].sum()
                    print("query time = {}".format(t))

    # 转换输出
    # rows = []
    # for threshold in level["thresholds"]:
    #     for num_part in level["num_parts"]:
    #         for num_perm in level["num_perms"]:
    #             print("Running LSH Ensemble benchmark "
    #                     "threshold = {}, num_part = {}, num_perm = {}".format(
    #                         threshold, num_part, num_perm))
    #             precisions, recalls, probe_times, results = benchmark_lshensemble(
    #                     threshold, num_perm, num_part, level["m"], storage_config,
    #                     index_data, query_data)
    #             for probe_time, precision, recall, result, query_set, query_key in zip(\
    #                     probe_times, precisions, recalls, results,\
    #                     query_data[1], query_data[2]):
    #                 rows.append((query_key, len(query_set), threshold,
    #                     num_part, num_perm, probe_time, precision, recall, result))
    # df_match = pd.DataFrame.from_records(rows,
    #     columns=["query_table", "candidate_table", "query_col_name", "candidate_col_name"])
    # df_match.to_csv("match.csv")
