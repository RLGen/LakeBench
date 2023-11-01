import numpy as np
import pandas as pd
import csv
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import PostgresDataLoader, CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from d3l.indexing.feature_extraction.values.glove_embedding_transformer import GloveTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer



# CSV data loader
dataloader = CSVDataLoader(
    root_path='/home/wangyanzhang/split1_query', #这个要改
    sep=','
)

print("LOADED!")


#  load them from disk if they have been created
name_index = unpickle_python_object('./name_webtable_large_split1&query.lsh')
format_index = unpickle_python_object('./format_webtable_large_split1&query.lsh')
value_index = unpickle_python_object('./value_webtable_large_split1&query.lsh')              #这些要改
embedding_index = unpickle_python_object('./embedding_webtable_large_split1&query.lsh')
distribution_index = unpickle_python_object('./distribution_webtable_large_split1&query.lsh')

print("Index LOADED!")

print("Start Querying!")
qe = QueryEngine(name_index, format_index, value_index, embedding_index, distribution_index)
table_name='csvData7001368__0'  #这个要改（查询表表名）
results, extended_results = qe.table_query(table=dataloader.read_table(table_name='csvData7001368__0'),
                                           aggregator=None, k=10, verbose=True)




for result in extended_results:
    query_table = table_name
    candidate_table = result[0]

    for x, column_info in enumerate(result[1]):
        column_info_name = f"column_info{x+1}"
        globals()[column_info_name] = column_info[0]

        #print(column_info[0])  # 输出 column_info[x][0]
        #print(column_info[1])  # 输出 column_info[x][1]
        print(f"{query_table}, {candidate_table}.csv, {column_info[0][0]}, {column_info[0][1]}")




file_path = '/home/wangyanzhang/d3l-main/d3l-main/examples/notebooks/results.csv'  #过渡用的查询结果表

with open(file_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)


     # Write the header
    writer.writerow(['query_table', 'candidate_table', 'query_col_name', 'candidate_col_name'])

    for result in extended_results:
        query_table = table_name
        candidate_table = result[0]

        for x, column_info in enumerate(result[1]):
            column_info_name = f"column_info{x+1}"
            globals()[column_info_name] = column_info[0]

            row = [f"{query_table}.csv", f"{candidate_table}.csv", column_info[0][0], column_info[0][1]]
            writer.writerow(row)
print("CSV file has been created successfully.")


# 读取包含真实候选表的CSV文件
groundtruth_file = "/data_ssd/webtable/small_query/ground_truth.csv"
groundtruth_df = pd.read_csv(groundtruth_file, usecols=["query_table", "candidate_table"])

# 读取查询到的候选表的CSV文件
results_file = "/home/wangyanzhang/d3l-main/d3l-main/examples/notebooks/results.csv"
results_df = pd.read_csv(results_file, usecols=["candidate_table"])

#print(table_name)
# 现有的查询表名
table_name = table_name+".csv"
print(table_name)
# 在真实候选表中找到对应的候选表集合Tq，并去重
Tq = set(groundtruth_df[groundtruth_df["query_table"] == table_name]["candidate_table"].unique())

# 在查询到的候选表中找到对应的候选表集合Tq'，并去重
Tq_prime = set(results_df["candidate_table"].unique())
print("真实候选表集合Tq:")
print(Tq)
print("查询到的候选表集合Tq':")
print(Tq_prime)
# 计算准确率和召回率
precision = len(Tq.intersection(Tq_prime)) / len(Tq_prime)
print("准确率:", precision)
recall = len(Tq.intersection(Tq_prime)) / len(Tq)

#print("准确率:", precision)
print("召回率:", recall)