import pickle
import os
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100000000  # 适当调整字段大小限制
import psutil
import csv

csv.field_size_limit(100000000)
import os
import time
from tqdm import tqdm



def createRawTokens(cpath, save_root):
    print("hi c")
    rawTokens_path = os.path.join(save_root, "rawTokens.csv")
    map_path = os.path.join(save_root, "setMap.pkl")
    # if not os.path.exists(rawTokens_path):
    #       os.makedirs(rawTokens_path)
    # if not os.path.exists(map_path):
    #       os.makedirs(map_path)
    t1=time.time()
    # 初始化
    setID = 0

    extracted_data = []
    setMap = {}
    table_names=os.listdir(cpath)
    i=0
    for table_name in tqdm(table_names):
        table_path = os.path.join(cpath, table_name)
        df = pd.read_csv(table_path,engine='python')
        i+=1
        for column_name in df.columns:
            raw_tokens = list(set(df[column_name].tolist()))
            pos = 0
            # 将每个集合和对应的SetID,pos保存到列表中
            for token in raw_tokens:
                if type(token) == str and not token.isdigit():
                    pos += 1
                    extracted_data.append((token, setID, pos))
            if pos != 0:
                setID += 1
                setMap[setID] = {'table_name': table_name, "column_name": column_name}
                                                                                                                                                                                                                                                                                                                           
    t=time.time()
    d= (t - t1) / 60
    print(f"读入时间:{d:.2f}分钟")   
    extracted_df = pd.DataFrame(extracted_data, columns=["RawToken", "SetID", "Position"])
    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024/1024))
    extracted_df.to_csv(rawTokens_path, index=False)  # 替换为你要保存的CSV文件路径
    
    with open(map_path, "wb") as tf:
        pickle.dump(setMap, tf)
    del extracted_data, df

if __name__ == '__main__':
    createRawTokens()


