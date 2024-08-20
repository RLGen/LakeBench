# import csv
# import pickle

# def convert_dict_to_csv(data_dict, output_file):
#     with open(output_file, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)

#         # 写入 CSV 文件的表头
#         #csv_writer.writerow(['Key', 'Value'])

#         # 遍历字典的键值对，写入 CSV 文件中
#         for key, value_list in data_dict.items():
#             for value in value_list:
#                 csv_writer.writerow([key, value])

# with open("../groundtruth/tusUnionBenchmark.pickle", 'rb') as f:
#     my_dict = pickle.load(f)
#     # 调用函数将字典转换为 CSV 格式的文件
#     convert_dict_to_csv(my_dict, '../groundtruth/tus_groundtruth.csv')
import pandas as pd
import shutil
import os

def copy_unique_files(source_folder, csv_file, destination_folder):
    # 读取 CSV 文件并获取唯一的文件名列表
    df = pd.read_csv(csv_file)
    unique_files = df['query_table'].unique()

    # 复制符合条件的文件到目标文件夹
    for filename in unique_files:
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        shutil.copy(source_file, destination_file)

# 示例参数
source_folder = 'benchmark/tus_benchmark/datalake'     # 原始文件所在的文件夹
csv_file = 'groundtruth/tus_groundtruth.csv'              # 包含文件名的 CSV 文件
destination_folder = 'benchmark/tus_benchmark/query'  # 目标文件夹

# 执行复制操作
copy_unique_files(source_folder, csv_file, destination_folder)



