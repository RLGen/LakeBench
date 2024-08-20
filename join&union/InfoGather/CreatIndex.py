import math
import pickle
import sys
import time

import numpy as np
import pandas as pd
import collections
import os
from tqdm import *


def CreatDataIndex(datalist,indexstorepath):

    # index KIA {attribute name: docid list}
    KIA = collections.defaultdict(list)

    # index of KIV {docid : terms list}
    KIV = {}

    # {docid: tablename#columnname}
    docNo =  collections.defaultdict(str)

    # {tablefilename: docid list} 
    docColumns =  collections.defaultdict(list)

    # {term:[docid .....]}
    inversed_index = collections.defaultdict(list)

    # docnumber
    doc_number = 0

    # ----------------------------------计算索引-------------------------------------

    index_start_time = time.time()

    for datafile in datalist:
        for root, dirs, files in os.walk(datafile):
            for filename in tqdm(files):
                if filename == "small_join.csv":
                    continue
                if filename.endswith(".csv"):
                    file_path = os.path.join(root, filename)
                    try:
                        df = pd.read_csv(file_path,encoding='utf-8',low_memory= False)
                    except Exception as e:
                        print("%s occur Error: %s" % (filename, e))
                        continue

                    headers = df.columns.tolist()
                    columns = {header: df[header].replace(np.nan, "NaN").unique() for header in headers}

                    for header, data in columns.items():
                        header = header.strip()
                        if not header:
                            continue
                        docName = f"{filename}#{header}".replace("\n", "")
                        docNo[doc_number] = docName

                        docColumns[filename].append(doc_number)

                        # create KIA index
                        if header:
                            KIA[header].append(doc_number)
                        else:
                            print("文件名转化失败，失败的属性名为：%s" % header)

                        # create inverted index
                        for term in data:
                            if type(term) ==str and not term.isdigit():
                                inversed_index[term].append(doc_number)

                        # create KIV index
                        KIV[doc_number] = data
                        doc_number += 1
    index_end_time = time.time()
    index_creat_time = index_end_time - index_start_time
    print("index creat time ",index_creat_time)

 

    ##-------------------------------------------存储索引-----------------------------------------------------
    path = os.path.join(indexstorepath,"KIA.pkl")
    stime = time.time()
    with open(path, "wb") as file:
        pickle.dump(KIA, file)
    etime = time.time()
    print(f"KIA index store success {etime - stime}")
    #del KIA

    path = os.path.join(indexstorepath, "docNo.pkl")
    stime = time.time()
    with open(path, "wb") as file:
        pickle.dump(docNo, file)
    etime = time.time()
    print(f"docNo index store success {etime - stime}")
    #del docNo

    path = os.path.join(indexstorepath, "inversed_index.pkl")
    stime = time.time()
    with open(path, "wb") as file:
        pickle.dump(inversed_index, file)
    etime = time.time()
    print(f"inversed_index index store success {etime - stime}")
    del inversed_index

    path = os.path.join(indexstorepath, "docColumns.pkl")
    with open(path, "wb") as file:
        pickle.dump(docColumns, file)
    #del docColumns
    print("docColumns index store success")

    path = os.path.join(indexstorepath, "KIV.pkl")
    stime = time.time()
    try:
        with open(path, "wb") as file:
            pickle.dump(KIV, file)
    except Exception as e:
        print(e)
    etime = time.time()
    print(f"KIV index store success {etime - stime}")

    path = os.path.join(indexstorepath, "doc_number.pkl")
    with open(path, "wb") as file:
        pickle.dump(doc_number, file)
    print(doc_number)
    print("索引创建完成")


















