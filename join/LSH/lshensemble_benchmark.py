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
from multiprocessing import Pool, cpu_count, Process, Queue
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct

from datasketch import MinHashLSHEnsemble, MinHash

rand_seed = 41
sample_ratio = 0.1
data_partition = 20
is_spark = False
is_mul = True
is_part = False
if is_part:
    data_partition = 1
storage_path = "LSH Ensemble"

query_tables = []
indexed_tables = []

query_path = "DataFilter/querytables"
# index_path = "DataFilter/csvdata"    
index_path = "/data/home/yuyanrui/Benchmark/DataFilter/csvdata"   

def convert_data_to_partition(all_data, parti_num):
    datas_list = [[] for _ in range(parti_num)]
    for table in all_data:
        datas_list[random.randint(0, data_partition-1)].append(table)
    return datas_list

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

# 将csv文件列表中每个表格中的column转换为三元组(minhashes, sets, keys)形式
# @profile
def bootstrap_sets(set_path, sets_files, num_perm):
    # 从csv中读取加载
    print("Creating sets...")
    print("Using num_perm = {}".format(num_perm))
    sizes = []
    keys = []
    # random.seed(rand_seed)
    minhashes = dict()
    ms = []
    for sets_file in sets_files:
        df = pd.read_csv(os.path.join(set_path, sets_file), dtype='str').dropna()
        columns = df.columns.tolist()
        data = df.values.T.tolist()
        for column,vals in zip(columns,data):
            # if random.random() > sample_ratio:
            #     continue
            # 需要对value去重
            vals = list(set(vals))
            # s = np.array(vals)
            # sets.append(s)
            # 域的键值
            keys.append(sets_file+"."+column)
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
def bootstrap_index_sets(set_path, sets_files, num_perm, threshold, num_part, m, storage_config):
    # 从csv中读取加载
    print("Creating sets...")
    print("Using num_perm = {}".format(num_perm))
    
    # 处理完整路径
    sets_files = [os.path.join(set_path, sets_file) for sets_file in sets_files]
    start = time.perf_counter()
    sizes = []
    keys = []
    lsh = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm,
            num_part=num_part, m=m, storage_config=storage_config)


    if is_spark:
        startr = time.perf_counter()
        # 创建 SparkSession
        spark = SparkSession.builder \
            .appName("CSV Column Distinct Values") \
            .getOrCreate()

        spark.conf.set("spark.sql.hive.filesourcePartitionFileCacheSize", "1024000000")

        # 读取 CSV 文件，并逐个处理
        for file in sets_files:
            # df = pd.read_csv(file, dtype='str').dropna()
            # df = spark.createDataFrame(df)
            # df = spark.read.load(file, format="csv", sep=",", inferSchema="true", header="true")
            df = spark.read.csv(file)
            # 获取第一行
            first_row = df.head(1)
            first_row_df = spark.createDataFrame(first_row, df.columns)
            # 除去第一行
            df = df.exceptAll(first_row_df)
            for column in df.columns:
                distinct_count = df.agg(countDistinct(col('`'+column+'`'))).collect()[0][0]
                sizes.append(distinct_count)
                if len(sizes)%10000==0:
                    sys.stdout.write("\rRead and sized {} sets".format(len(sizes)))
        sys.stdout.write("\n")
        print("spark size time = {}".format(time.perf_counter() - startr))

    else:
        startr = time.perf_counter()
        for sets_file in sets_files:
            df = pd.read_csv(sets_file, dtype='str').dropna()
            columns = df.columns.tolist()
            data = df.values.T.tolist()
            for column,vals in zip(columns,data):
                # 需要对value去重
                vals = list(set(vals))
                sizes.append(len(vals))
                if len(sizes)%10000==0:
                    sys.stdout.write("\rRead and sized {} sets".format(len(sizes)))
        sys.stdout.write("\n")
        print("size time = {}".format(time.perf_counter() - startr))
    lsh.count_partition(sizes)

    ind = 0
    for sets_file in sets_files:
        if is_spark:
            # df = pd.read_csv(sets_file, dtype='str').dropna()
            # df = spark.createDataFrame(df)
            df = spark.read.csv(sets_file)
            # 获取第一行
            first_row = df.head(1)
            first_row_df = spark.createDataFrame(first_row, df.columns)
            # 除去第一行
            df = df.exceptAll(first_row_df)
            # 遍历每个列，进行去重和 MinHash 索引
            for column in df.columns:
                # 去重，并将列中的值收集到一个列表中
                key = sets_file+"."+column
                keys.append(key)
                distinct_values = df.select('`'+column+'`').distinct().rdd.flatMap(lambda x: x).collect()

                # 创建 MinHash 对象，并传入去重后的值
                minhash = MinHash(num_perm, hashfunc=_hash_32)
                for value in distinct_values:
                    minhash.update(str(value))
                lsh.index((key, minhash, sizes[ind]))
                if len(keys)%10000==0:
                    sys.stdout.write("\rRead and minhashed {} sets".format(len(keys)))
                ind += 1
        else:
            df = pd.read_csv(sets_file, dtype='str').dropna()
            columns = df.columns.tolist()
            data = df.values.T.tolist()
            for column,vals in zip(columns,data):
                # 需要对value去重
                vals = list(set(vals))
                # 域的键值
                key = sets_file+"."+column
                keys.append(key)
                # 生成minhash
                mh = MinHash(num_perm, hashfunc=_hash_32)
                for word in vals:
                    mh.update(str(word))
                lsh.index((key, mh, sizes[ind]))
                if len(keys)%10000==0:
                    sys.stdout.write("\rRead and minhashed {} sets".format(len(keys)))
                ind += 1
    sys.stdout.write("\n")
    
    print("build index time = {}, size = {} bytes".format(time.perf_counter() - start, asizeof.asizeof(lsh)))
    
    if is_spark:
        spark.stop()

    return (lsh, sizes, keys)

# 获取query column在groundtruth表格中的对应结果
def get_reference(df, column_name):
    filtered_df = df[df['query_table'] + '.' + df['query_col_name'] == column_name]
    reference_df = filtered_df['candidate_table'] + '.' + filtered_df['candidate_col_name']
    return reference_df.tolist()

# 对lshensemble进行测试 返回时间与查询结果
def benchmark_lshensemble(threshold, num_perm, num_part, m, storage_config,
        index_data, query_data):
    # print("Building LSH Ensemble index")
    # (lsh, indexed_sizes, keys) = index_data
    lsh = index_data[0]
    # start = time.perf_counter()
    # lsh = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm,
    #         num_part=num_part, m=m, storage_config=storage_config)
    # lsh.index((key, minhash, size)
    #         for key, minhash, size in \
    #                 zip(keys, minhashes[num_perm], indexed_sizes))
    # print("build index time = {}, size = {} bytes".format(time.perf_counter() - start, asizeof.asizeof(lsh)))
    print("Querying")
    df = pd.read_csv('data/att_groundtruth.csv', dtype='str').dropna()
    (minhashes, sizes, keys) = query_data
    probe_times = []
    precisions = []
    recalls = []
    results = []
    match_results = []
    for qsize, minhash, key in zip(sizes, minhashes[num_perm], keys):
        # 记录查询时间
        start = time.perf_counter()
        result = list(lsh.query(minhash, qsize))
        probe_times.append(time.perf_counter() - start)
        results.append(result)
        match_results.append((key, result))
        # 记录精度召回
        reference = get_reference(df, key)
        precision, recall = get_precision_recall(result, reference)
        precisions.append(precision)
        recalls.append(recall)
        sys.stdout.write("\rQueried {} sets".format(len(probe_times)))
        # if len(probe_times)>5:
        #     break
    sys.stdout.write("\n")
    return precisions, recalls, probe_times, results, match_results


def _compute_containment(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    intersection = len(np.intersect1d(x, y, assume_unique=True))
    return float(intersection) / float(len(x))


levels = {
    "test": {
        "thresholds": [1.0],
        "num_parts": [8],
        "num_perms": [256],
        "m": 4,
    },
    "test2": {
        "thresholds": [1.0],
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
            default=os.path.join(storage_path, "lshensemble_benchmark_query_results.csv"))
    parser.add_argument("--ground-truth-results", type=str,
            default="lshensemble_benchmark_ground_truth_results.csv")
    parser.add_argument("--indexed-sets-sample-ratio", type=float, default=1)
    parser.add_argument("--level", type=str, choices=levels.keys(), 
            default="test")
    parser.add_argument("--use-redis", action="store_true")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args(sys.argv[1:])

    level = levels[args.level]
    
    index_data, query_data = None, None
    index_data_cache = os.path.join(storage_path, "{}.pickle".format("index"))
    query_data_cache = os.path.join(storage_path, "{}.pickle".format("query"))

    # 随机取100张query表
    # df = pd.read_csv("data/att_groundtruth.csv", dtype='str').dropna()
    # column_data = df["query_table"]
    # unique_values = column_data.unique()
    # np.random.seed(rand_seed)
    # random_values = np.random.choice(unique_values, size=100, replace=False)
    # query_tables = random_values.tolist()
    
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

    # # 将querytables中所有表作为index表
    # for filename in os.listdir(query_path):
    #     if filename.endswith(".csv") and "groundtruth" not in filename:
    #         query_tables.append(filename)

    # # 查询数据缓存
    # if os.path.exists(query_data_cache):
    #     print("Using cached query sets {}".format(query_data_cache))
    #     with open(query_data_cache, "rb") as d:
    #         query_data = pickle.load(d)
    # else:
    #     # print("Using query sets {}".format(query_tables))
    #     start = time.perf_counter()
    #     query_data = bootstrap_sets(query_path, query_tables, num_perm=level["num_perms"][0]) #只执行第一个设定的num_perms值
    #     print("minhash query_data time = {}".format(time.perf_counter() - start))
    #     with open(query_data_cache, "wb") as d:
    #         pickle.dump(query_data, d)
            
    # 将csvdata中所有表作为index表
    random.seed(rand_seed)
    if is_part:
        for filename in os.listdir(index_path):
            if filename.endswith(".csv") and "groundtruth" not in filename:
                if random.random() > sample_ratio:
                    continue
                indexed_tables.append(filename)
        indexed_tables = [indexed_tables]
    else:
        for filename in os.listdir(index_path):
            if filename.endswith(".csv") and "groundtruth" not in filename:
                indexed_tables.append(filename)
        indexed_tables = convert_data_to_partition(indexed_tables, data_partition)
    
    # 索引数据缓存 
    for id in range(data_partition):
        index_data_cache_name = index_data_cache+"-"+str(id)
        if os.path.exists(index_data_cache_name):
            print("Using cached indexed sets {}".format(index_data_cache_name))
            with open(index_data_cache_name, "rb") as d:
                index_data = pickle.load(d)
            # pdb.set_trace()
        else:
            # print("Using indexed sets {}".format(indexed_tables))
            start = time.perf_counter()
            index_data = bootstrap_index_sets(
                index_path, indexed_tables[id], num_perm=level["num_perms"][0], 
                threshold=level["thresholds"][0], num_part=level["num_parts"][0], m=level["m"], storage_config=storage_config)
            print("minhash index_data time = {}".format(time.perf_counter() - start))
            with open(index_data_cache_name, "wb") as d:
                pickle.dump(index_data, d)
        pdb.set_trace()
    
    
    # 输出匹配 同att_groundtruth格式
    match_filename = os.path.join(storage_path, "match.csv")

    # 运行lshensamble，获取相关指标，整理结果并保存
    if os.path.exists(args.query_results):
        df = pd.read_csv(args.query_results).dropna()
    else:
        rows = []
        for threshold in level["thresholds"]:
            for num_part in level["num_parts"]:
                for num_perm in level["num_perms"]:
                    print("Running LSH Ensemble benchmark "
                            "threshold = {}, num_part = {}, num_perm = {}".format(
                                threshold, num_part, num_perm))
                    precisions, recalls, probe_times, results, match_results = benchmark_lshensemble(
                            threshold, num_perm, num_part, level["m"], storage_config, 
                            index_data, query_data)
                            
                    match_rows = []
                    for row in match_results:
                        query_table = row[0].split(".csv.")[0] + ".csv"
                        query_col_name = row[0].split(".csv.")[1]
                        for candidate in row[1]:
                            match_rows.append((query_table,
                            candidate.split(".csv.")[0] + ".csv",
                            query_col_name, 
                            candidate.split(".csv.")[1]))
                    df_match = pd.DataFrame.from_records(match_rows,
                        columns=["query_table", "candidate_table", "query_col_name", "candidate_col_name"])
                    df_match.to_csv(str(threshold)+"-"+str(num_part)+"-"+str(num_perm)+"-"+"match.csv", index=False)

                    for probe_time, precision, recall, result, query_size, query_key in zip(\
                            probe_times, precisions, recalls, results,\
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
                t = sub["query_time"].quantile(0.9)
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
