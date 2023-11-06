
import math
import time
from tqdm import tqdm
import pandas as pd
import json
from dataProcess import *
from josie import *
import pickle
from heap import *

def api(qpath: str,
        save_root: str, 
        result_root: str,
        k:int):
    
    # 读取文件
    with open("results_web/setMap.pkl", "rb") as tf:
        setMap = pickle.load(tf)
    outpath = os.path.join(save_root, "outputs")
    tf = open(outpath + "integerSet.json", "r")
    integerSet = json.load(tf)
    tf = open(outpath + "PLs.json", "r")
    PLs = json.load(tf)
    print("PLs")
    tf = open(outpath+"rawDict.json", "r")
    rawDict= json.load(tf)
    print("rawDict")
    tf.close()
    table_names = os.listdir(qpath)
    
    print("load suc!")

    durs = []
    res=[]
    i=0
    for table_name in tqdm(table_names):
        i+=1
        # try:
        ignore=False
        query_ID = -1
        table_path = os.path.join(qpath, table_name)
        df = pd.read_csv(table_path)
        for column_name in df.columns:
            query_ID=readQueryID(setMap,table_name,column_name)
            if  query_ID>0:
                ignore = True
            raw_tokens = list(set(df[column_name].tolist()))
            t1 = time.time()
            result=searchMergeProbeCostModelGreedy(integerSet, PLs, raw_tokens,rawDict,setMap, k, ignore,query_ID)
            print(result)
            t2 = time.time()
            if result!=0:
                dur = (t2 - t1)
                durs.append(dur) 
                # print(f"在线处理一个query时间：{dur:.2f}秒,query长度：{len(raw_tokens)}")            
                for x in result:
                    # print(table_name,setMap[x+1]["table_name"],column_name,setMap[x+1]["column_name"])
                    re=[table_name,setMap[x+1]["table_name"],column_name,setMap[x+1]["column_name"]]
                    res.append(re)
    mean = np.mean(durs)
    print(f"平均时间：{mean:.2f}秒")
    df_end = pd.DataFrame(res, columns=['query_table','candidate_table','query_column','candidate_column'])
    result_path=os.path.join(result_root, "join_top"+k+".csv")
    df_end.to_csv(result_path, index=False)