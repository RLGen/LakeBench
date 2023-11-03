import os

path_a = '/data/opendata/large/query'
path_b = "/d3l-main/d3l-main/examples/notebooks/opendata_large_60"

files_in_path_a = [f for f in os.listdir(path_a) if f.endswith('.csv')]
files_in_path_b = [f for f in os.listdir(path_b) if f.endswith('.csv')]

files_only_in_path_a = [file for file in files_in_path_a if file not in files_in_path_b]

print("CSV file in path A but not in path B:")
for file_name in files_only_in_path_a:
    print(file_name)
