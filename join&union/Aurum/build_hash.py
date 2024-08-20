import logging

import numpy as np
import csv
import os
import pickle
import time
from datasketch import MinHash
from multiprocessing import Process, Queue
from tqdm import tqdm
import sys
#####
import pandas as pd
import multiprocessing
# 设置读取CSV的字段大小
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="webtable")

hp = parser.parse_args()

# Change the data paths to where the benchmarks are stored
if "webtable" in hp.task:
    target_file_sets = 'data/webtable/small/datalake'
    if hp.task == "webtable_large":
        target_file_sets = 'data/webtable/large/datalake'
elif "opendata" in hp.task:
    target_file_sets = 'data/opendata/small/benchmark'
    if hp.task == "opendata_large":
        target_file_sets = 'data/opendata/large/datalake'

csv.field_size_limit(sys.maxsize)


hnsw = []

def build_hash(file_ls, queue, queue_hnsw,idx):
    all_file_target = []
    data_embadding = []
    # 将所有.csv文件的路径存入数组
    for j in range(0, len(file_ls)):
        if file_ls[j].endswith(".csv"):
            all_file_target.append(file_ls[j])

    head_array = []
    # 建立索引
    for i in range(0, len(all_file_target)):
        # 读入数据
        data_array_target = []
        # print(asizeof.asizeof(data_array_target))
        with open(all_file_target[i], encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            flag = 0
            for idx, row in enumerate(csv_reader):
                if flag == 0:
                    head_array = row
                    flag = 1
                elif row:
                    data_array_target.append(row)

        str0 = all_file_target[i].split("\\")[-1]
        str0 = str0[:-4]
        # print(str0)
        table_temp = []
        # 对每列内容建立minhash索引
        for j in range(0, len(data_array_target[0])):
            m1 = MinHash(num_perm=128)
            temp = list(set([row[j] for row in data_array_target if j < len(row)]))
            for str_i in temp:
                m1.update(str_i.encode('utf-8'))
            # print(temp)
            table_temp.append(list(m1.hashvalues))
        temp_array = np.array(table_temp)
        data_embadding.append((str0, temp_array))
        queue.put(1)
    queue_hnsw.put(data_embadding)
    queue.put((-1, "test-pid"))
    # print("{} end".format(idx))


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


start_time = time.time()
file_ls = []
split_num = 30


for i in target_file_sets:
    for root, dirs, files in os.walk(i):
        # print(root)
        # print(files)
        # exit()
        root_file_ls = [os.path.join(root, file) for file in files]
        file_ls.extend(root_file_ls)
# 读入候选集目录

sub_file_ls = split_list(file_ls, split_num)

process_list = []
queue_hnsw = multiprocessing.Manager().Queue()

#####
# 为每个进程创建一个队列
queues = [multiprocessing.Manager().Queue() for i in range(split_num)]
# queue = Queue()
# 一个用于标识所有进程已结束的数组
finished = [False for i in range(split_num)]

# 为每个进程创建一个进度条
bars = [tqdm(total=len(sub_file_ls[i]), desc=f"bar-{i}", position=i) for i in range(split_num)]
# 用于保存每个进程的返回结果
results = [None for i in range(split_num)]

for i in range(split_num):
    process = Process(target=build_hash, args=(sub_file_ls[i], queues[i], queue_hnsw, i))
    process_list.append(process)
    process.start()

while True:
    for i in range(split_num):
        queue = queues[i]
        bar = bars[i]
        try:
            # 从队列中获取数据
            # 这里需要用非阻塞的get_nowait或get(True)
            # 如果用get()，当某个进程在某一次处理的时候花费较长时间的话，会把后面的进程的进度条阻塞着
            # 一定要try捕捉错误，get_nowait读不到数据时会跑出错误
            res = queue.get_nowait()
            if isinstance(res, tuple) and res[0] == -1:
                # 某个进程已经处理完毕
                finished[i] = True
                results[i] = res[1]
                continue
            bar.update(res)
        except Exception as e:
            continue

            # 所有进程处理完毕
    if all(finished):
        break


for process in process_list:
    process.join()


while not queue_hnsw.empty():
    try:
        k = queue_hnsw.get_nowait()
        hnsw.extend(k)
    except Exception as e:
        continue

import json

with open("./hnsw_output_" + hp.task + ".pkl", 'wb') as f:
    pickle.dump(hnsw, f)

end_time = time.time()
print("\n")
print(end_time - start_time)
print("\n")
# 计算时分秒
run_time = round(end_time - start_time)
hour = run_time // 3600
minute = (run_time - 3600 * hour) // 60
second = run_time - 3600 * hour - 60 * minute
# 输出
print(f'该程序运行时间：{hour}小时{minute}分钟{second}秒')
