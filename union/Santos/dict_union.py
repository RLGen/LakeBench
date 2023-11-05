# import glob
# import time
# import pickle #version 4
# import pandas as pd
import generalFunctions as genFunc
# import expandSearch as expand
# import csv
# import os

# def merge_dicts(dict1, path):
#     dict2 = genFunc.loadDictionaryFromPickleFile(path)
#     merged_dict = dict1.copy()

#     for key, value in dict2.items():
#         if key in merged_dict:
#             merged_dict[key].extend(value)
#         else:
#             merged_dict[key] = value

#     return merged_dict

# relation_path = '_main_relation_index.pickle'
# triple_path = '_main_triple_index.pickle'
# yago_path = '_main_yago_index.pickle'

# relation_dict = genFunc.loadDictionaryFromPickleFile('/data/hashmap/hashmap/opendata0'+relation_path)
# triple_dict = genFunc.loadDictionaryFromPickleFile('/data/hashmap/hashmap/opendata0'+triple_path)
# yago_dict = genFunc.loadDictionaryFromPickleFile('/data/hashmap/hashmap/opendata0'+yago_path)

# for i in range(1, 4):
#     relation_dict = merge_dicts(relation_dict, '/data/hashmap/hashmap/opendata'+ str(i) + relation_path)
#     triple_dict = merge_dicts(triple_dict, '/data/hashmap/hashmap/opendata'+ str(i) + triple_path)
#     yago_dict = merge_dicts(yago_dict, '/data/hashmap/hashmap/opendata'+ str(i) +yago_path)

# genFunc.saveDictionaryAsPickleFile(relation_dict, 'hashmap/opendata' + relation_path)
# genFunc.saveDictionaryAsPickleFile(triple_dict, 'hashmap/opendata' + triple_path)
# genFunc.saveDictionaryAsPickleFile(yago_dict, 'hashmap/opendata' + yago_path)

#写一个ground_truth转dict的函数，写一个造intentcol的函数
import pandas as pd

# 读取CSV文件，假设文件名为data.csv，第一列为querytable，第二列为candidatetable
# data = pd.read_csv('/data_ssd/opendata/small_query/ground_truth.csv')

# # 创建一个空字典用于存储结果
# table_dict = {}

# # 遍历每一行，将数据添加到字典中
# for index, row in data.iterrows():
#     query_table = row['query_table']
#     candidate_table = row['candidate_table']
    
#     if query_table in table_dict:
#         table_dict[query_table].append(candidate_table)
#     else:
#         table_dict[query_table] = [candidate_table]

# genFunc.saveDictionaryAsPickleFile(table_dict, 'groundtruth/opendataUnionBenchmark.pickle')
#print(table_dict)
#把querytable的名字取出来，然后设置字典为0

from pathlib import Path
import glob

QUERY_TABLE_PATH = '/data_ssd/opendata/small_query/query/'
query_table_file_name = glob.glob(QUERY_TABLE_PATH+"*.csv")
col_dict = {}
for table in query_table_file_name:
    stemmed_file_name = Path(table).stem
    col_dict[stemmed_file_name] = 0
genFunc.saveDictionaryAsPickleFile(col_dict, 'groundtruth/opendataIntentColumnBenchmark.pickle')