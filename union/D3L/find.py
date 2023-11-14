import os

# 输入两个文件夹路径
path_a = '/data/opendata/large/query'
path_b = "/home/wangyanzhang/d3l-main/d3l-main/examples/notebooks/opendata_large_60"

# 获取路径A和路径B中的CSV文件列表
files_in_path_a = [f for f in os.listdir(path_a) if f.endswith('.csv')]
files_in_path_b = [f for f in os.listdir(path_b) if f.endswith('.csv')]

# 在路径A中但不在路径B中的文件名
files_only_in_path_a = [file for file in files_in_path_a if file not in files_in_path_b]

# 打印结果
print("在路径A中但不在路径B中的CSV文件：")
for file_name in files_only_in_path_a:
    print(file_name)
