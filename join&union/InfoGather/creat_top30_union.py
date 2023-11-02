
import os
import pickle

import pandas as pd


def list_to_list_arry(key,values):
        res = []
        try:
            for value in values:
                res.append([key,value])
        except Exception as e:
            print(value)
            print(e)
        return res

def creat_top30(filepath,storepath):
    
    os.makedirs(storepath,exist_ok=True)
    with open(filepath,"rb") as f:
        data = pickle.load(f)

    for i in range(5,31,5):
        res_ = []
        filename = f"webtable_small_top{i}.csv"
        file = os.path.join(storepath,filename)
        for key,values in data.items():
            values_ = values[:i]
            ele_list = list_to_list_arry(key,values_)
            res_.extend(ele_list)

        df = pd.DataFrame(res_)
        df.to_csv(file,index=False, header=False)

filepath= "/data/lijiajun/infogather/webtables/index/unin_res.pkl"

storepath = "/data/lijiajun/infogather/webtables/topk/union"
os.makedirs(storepath,exist_ok=True)

creat_top30(filepath,storepath)


def creat_union_opendata_top60(filepath,storepath):
    

    with open(filepath,"rb") as f:
        data = pickle.load(f)

    for i in range(10,61,10):
        res_ = []
        filename = f"opendata_union_small_top{i}.csv"
        file = os.path.join(storepath,filename)
        for key,values in data.items():
            values_ = values[:i]
            ele_list = list_to_list_arry(key,values_)
            res_.extend(ele_list)

        df = pd.DataFrame(res_)
        df.to_csv(file,index=False, header=False)

# filepath = "/data/lijiajun/infogather/opendata/index/unin_res.pkl"
# storepath = "/data/lijiajun/infogather/opendata/union/topk"
# os.makedirs(storepath,exist_ok=True)
# creat_union_opendata_top60(filepath,storepath)







    

