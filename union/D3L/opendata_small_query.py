import numpy as np
import pandas as pd
import csv
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import PostgresDataLoader, CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from d3l.indexing.feature_extraction.values.glove_embedding_transformer import GloveTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

import os
import multiprocessing
from multiprocessing import Process, Queue
from tqdm import tqdm
import time

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

def sub_process(query_tables, queue):
    for i, table_name_with_extension in enumerate(query_tables):

        table_name = os.path.splitext(table_name_with_extension)[0]
        output_folder = "/home/wangyanzhang/d3l-main/d3l-main/examples/notebooks/opendata_large_60"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        results_file = os.path.join(output_folder, f"{table_name}.csv")

        if os.path.exists(results_file):
            print(f"跳过查询表 {i + 1}，因为结果文件已经存在：{results_file}")
            queue.put(1)  # 在队列中放入一个占位符值
            continue

        
        # 执行查询
        results, extended_results = qe.table_query(table=dataloader.read_table(table_name=table_name),
                                                aggregator=None, k=60, verbose=True)


        # 创建一个新的 CSV 文件
        with open(results_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(['query_table', 'candidate_table', 'query_col_name', 'candidate_col_name'])

            # 写入查询结果到 CSV 文件
            for result in extended_results:
                query_table = table_name
                candidate_table = os.path.basename(result[0])

                for x, column_info in enumerate(result[1]):
                    column_info_name = f"column_info{x+1}"
                    globals()[column_info_name] = column_info[0]

                    row = [f"{query_table}.csv", f"{candidate_table}.csv", column_info[0][0], column_info[0][1]]
                    writer.writerow(row)

        print(f"Results for query table {i + 1} have been written to {results_file}")
        queue.put(1)
    queue.put((-1, "test-pid"))

if __name__ == "__main__":
    # 记录程序开始时间
    start_time = time.time()

   # CSV data loader
    dataloader = CSVDataLoader(
        root_path='/data/opendata/large/query/', #这个要改
        sep=','
    )
    print("LOADED!")

    #  load them from disk if they have been created
    name_index = unpickle_python_object('./name_opendata_large.lsh')
    print("Name Unpickled!")
    format_index = unpickle_python_object('./format_opendata_large.lsh')
    print("Format Unpickled!")
    value_index = unpickle_python_object('./value_opendata_large.lsh')              #这些要改
    print("Value Unpickled!")
    embedding_index = unpickle_python_object('./embedding_opendata_large.lsh')
    print("Embedding Unpickled!")
    distribution_index = unpickle_python_object('./distribution_opendata_large.lsh')
    print("Distribution Unpickled!")
    print("Index LOADED!")

    split_num = 40
    # 指定包含查询表的文件夹路径
    query_tables_folder = "/data/opendata/large/query/"

    # 获取文件夹中的所有文件名
    query_tables = [f for f in os.listdir(query_tables_folder) if f.endswith(".csv")]
    # query_tables = [
    # "USA_CSV0000000000001722.csv",
    # "USA_CSV0000000000033169__0.csv",
    # "USA_CSV0000000000035902__29.csv",
    # "CAN_CSV0000000000002115__7.csv",
    # "UK_CSV0000000000007304__15.csv",
    # "UK_CSV0000000000005907__2.csv",
    # "UK_CSV0000000000005907__37.csv",
    # "UK_CSV0000000000008169__5.csv",
    # "USA_CSV0000000000033989.csv",
    # "SG_CSV0000000000000040__7.csv",
    # "CAN_CSV0000000000027355__0.csv",
    # "USA_CSV0000000000033678__3.csv",
    # "UK_CSV0000000000002064__13.csv",
    # "CAN_CSV0000000000002115__27.csv",
    # "UK_CSV0000000000008767.csv",
    # "USA_CSV0000000000034235__12.csv",
    # "UK_CSV0000000000008169__2.csv",
    # "CAN_CSV0000000000027370__4.csv",
    # "UK_CSV0000000000005854__0.csv",
    # "USA_CSV0000000000034882__33.csv",
    # "UK_CSV0000000000005907__9.csv",
    # "USA_CSV0000000000001106.csv",
    # "SG_CSV0000000000001394.csv",
    # "USA_CSV0000000000036852__0.csv",
    # "UK_CSV0000000000011359__15.csv",
    # "CAN_CSV0000000000001708__5.csv",
    # "CAN_CSV0000000000001471__1.csv",
    # "CAN_CSV0000000000027354__10.csv",
    # "USA_CSV0000000000037169__13.csv",
    # "USA_CSV0000000000034882__31.csv",
    # "USA_CSV0000000000035085__13.csv",
    # "USA_CSV0000000000001657.csv",
    # "UK_CSV0000000000005605__15.csv",
    # "CAN_CSV0000000000001724__16.csv",
    # "USA_CSV0000000000035412__2.csv",
    # "CAN_CSV0000000000027243.csv",
    # "CAN_CSV0000000000027338__9.csv",
    # "USA_CSV0000000000035554__27.csv",
    # "USA_CSV0000000000035350.csv",
    # "USA_CSV0000000000034543__6.csv",
    # "CAN_CSV0000000000001294__0.csv",
    # "CAN_CSV0000000000027353__9.csv",
    # "USA_CSV0000000000034123.csv",
    # "CAN_CSV0000000000001780__20.csv",
    # "USA_CSV0000000000000999.csv",
    # "CAN_CSV0000000000001358__24.csv",
    # "USA_CSV0000000000033847__15.csv",
    # "USA_CSV0000000000007166.csv",
    # "UK_CSV0000000000007304__0.csv",
    # "CAN_CSV0000000000010611.csv",
    # "UK_CSV0000000000002010__2.csv",
    # "UK_CSV0000000000005907__23.csv",
    # "USA_CSV0000000000033339__34.csv",
    # "USA_CSV0000000000033533__2.csv"
    # ]
    # query_tables = [
    # "USA_CSV0000000000001722.csv",
    # "USA_CSV0000000000033169__0.csv",
    # "USA_CSV0000000000035350.csv"
    # ]


    qe = QueryEngine(name_index, format_index, value_index, embedding_index, distribution_index)

    sub_file_ls = split_list(query_tables, split_num)
    process_list = []

    # 为每个进程创建一个队列
    queues = [Queue() for i in range(split_num)]
    finished = [False for i in range(split_num)]

    # 为每个进程创建一个进度条
    bars = [tqdm(total=len(sub_file_ls[i]), desc=f"bar-{i}", position=i) for i in range(split_num)]
    results = [None for i in range(split_num)]

    for i in range(split_num):
        process = Process(target=sub_process, args=(sub_file_ls[i], queues[i]))
        process_list.append(process)
        process.start()

    while True:
        for i in range(split_num):
            queue = queues[i]
            bar = bars[i]
            try:
                res = queue.get_nowait()
                if isinstance(res, tuple) and res[0] == -1:
                    finished[i] = True
                    results[i] = res[1]
                    continue
                bar.update(res)
            except Exception as e:
                continue

        if all(finished):
            break

    for process in process_list:
        process.join()

    # 记录程序结束时间
    end_time = time.time()

    # 计算运行时间（以秒为单位）
    run_time = end_time - start_time

    # 打印运行时间
    print(f"程序运行时间：{run_time} 秒")


