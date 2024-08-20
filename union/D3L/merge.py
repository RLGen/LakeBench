import pandas as pd
import glob

# 获取要合并的CSV文件列表（按存放顺序返回）
csv_files = glob.glob("/home/wangyanzhang/d3l-main/d3l-main/examples/notebooks/opendata_large_60/*.csv")  # 将路径替换为包含CSV文件的文件夹路径

# 创建一个空的DataFrame来存储合并后的数据
combined_data = pd.DataFrame()

# 逐个读取CSV文件并将其合并到combined_data中
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    combined_data = pd.concat([combined_data, df], ignore_index=True)

# 将合并后的数据保存为一个新的CSV文件
combined_data.to_csv("/home/wangyanzhang/d3l-main/d3l-main/examples/notebooks/opendata_large_60.csv", index=False)  # 将文件名替换为您想要的文件名

