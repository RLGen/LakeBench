import psutil
import numpy as np
import os
import time
import pandas as pd
import hashlib
import json
from tqdm import tqdm


def isEqual(x,y):
    if x[2]==y[2] and x[3]==y[3]:
        return True
    else:
        return False

def createTokenID(save_root):
    rawTokens_path = os.path.join(save_root, "rawTokens.csv")
    outpath = os.path.join(save_root, "outputs")
    # if not os.path.exists(outpath):
    #       os.makedirs(outpath)    
    RawTokens_df=pd.read_csv(rawTokens_path)

    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------------1.分组初步创建PL----------------------------------------------
    t1=time.time()
    PL_list=[]
  
    grouped = RawTokens_df[['RawToken', 'SetID']].groupby('RawToken')

    ###单进程
    i=0
    for group_name, group_df in tqdm(grouped):
        token_data = group_df["SetID"].values.tolist()
        PL_list.append([group_name, token_data, len(token_data), hashlib.sha256(str(token_data).encode()).hexdigest()])
        i+=1
    t2 = time.time()
    d1 = (t2 - t1) / 60

    print(f"创建PL的时间:{d1:.2f}分钟")
    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024/1024))
    # 2.排序生成tokenID
    PL_list_sorted = sorted(PL_list, key=lambda x: (x[2], x[3]), reverse=False)
    del PL_list
    tokenIDs = []
    for i in range(len(PL_list_sorted)):
        tokenIDs.append([PL_list_sorted[i][0], int(i)])

 #  3.生成GIDs

    gids = [0] * len(PL_list_sorted)
    groudID = 0
    for i in range(len(PL_list_sorted)):
        gids[i] = groudID
        if i == len(PL_list_sorted) - 1:
            break
        if not isEqual(PL_list_sorted[i], PL_list_sorted[i + 1]):
            groudID = groudID + 1
    del PL_list_sorted

    # 4.生成tokenTable
    tokenTable = []
    for i in range(len(tokenIDs)):

        x = tokenIDs[i] + [gids[i]]
        tokenTable.append(x)
    del tokenIDs
    del gids
    tokenTable_df = pd.DataFrame(tokenTable, columns=["RawToken", "TokenID", "GroupID"])
    del tokenTable
    t3 = time.time()
    d2 = (t3 - t2) / 60
    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024/1024))
    print(f"创建tokenTable的时间:{d2:.2f}分钟")

    # 5.生成integerSet

    integerSet = {}
    merged_df = pd.merge(RawTokens_df, tokenTable_df, on='RawToken', how='left')
    del RawTokens_df, tokenTable_df

    setLen = {}
    grouped1 = merged_df.groupby('SetID')
    for group_name, group_df in grouped1:
        tids = group_df["TokenID"].values.tolist()
        integerSet[group_name] = tids
        setLen[group_name] = len(tids)
    t4 = time.time()
    d3 = (t4 - t3) / 60
    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024/1024))
    print(f"创建integerSet的时间:{d3:.2f}分钟")
    tf = open(os.path.join(outpath, "integerSet.json"), "w")
    json.dump(integerSet, tf)
    tf.close()

    PLs = {}
    rawDict = {}
    grouped2 = merged_df.groupby('TokenID')
    for group_name, group_df in tqdm(grouped2):
        setIDs = group_df["SetID"].values.tolist()
        positions=group_df["Position"].values.tolist()
        gid =int(group_df["GroupID"].values.tolist()[0])
        raw=group_df["RawToken"].values.tolist()[0]
        tid=int(group_name)
        freq=len(setIDs)
        rawDict[raw] =[tid,gid,freq] #tid,gid,freqs
        PL = []
        for i in range(freq):
            PL.append((setIDs[i],positions[i],setLen[setIDs[i]]))
        PLs[tid]=PL

    tf = open(os.path.join(outpath, "PLs.json"), "w")
    json.dump(PLs, tf)
    tf.close()
    tf1 = open(os.path.join(outpath, "rawDict.json"), "w")
    json.dump(rawDict, tf1)
    tf1.close()

    t5 = time.time()
    d4 = (t5 - t1) / 60
    process = psutil.Process(os.getpid())
    print("当前内存占用：%s" % (process.memory_info().rss / 1024 / 1024/1024))
    print(f"创建倒排索引的总时间:{d4:.2f}分钟")




if __name__ == '__main__':

    createTokenID()