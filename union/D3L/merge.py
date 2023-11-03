import pandas as pd
import glob

csv_files = glob.glob("/d3l-main/d3l-main/examples/notebooks/opendata_large_60/*.csv")  # 将路径替换为包含CSV文件的文件夹路径

combined_data = pd.DataFrame()

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    combined_data = pd.concat([combined_data, df], ignore_index=True)

combined_data.to_csv("/d3l-main/d3l-main/examples/notebooks/opendata_large_60.csv", index=False)  # 将文件名替换为您想要的文件名

