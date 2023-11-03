import numpy as np
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.nn.parallel import DataParallel
import logging
from datetime import datetime
import os
import csv
import pickle
import multiprocessing


def process_onedataset(dataset_file,model_name ='output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27',
                    storepath = "/home/lijiajun/deepjoin/webtable/final_result/"):

    path,filename_dataset = os.path.split(dataset_file)

    model = SentenceTransformer(model_name)
    os.makedirs(storepath,exist_ok=True)
    storedata = []
    if os.path.isfile(dataset_file):
        print("process data",dataset_file)
        try:
            #记载数据
            with open(dataset_file,"rb") as f:
                data = pickle.load(f)
            for ele in tqdm(data):
                key,value = ele
                sentence_embeddings = model.encode(value)
                sentence_embeddings_np = np.array(sentence_embeddings)
                tu1 = (key,sentence_embeddings_np)
                storedata.append(tu1)
        except Exception as e:
            print(e)
    storefilename = os.path.join(storepath,filename_dataset)
    with open(storefilename,"wb") as f:
        pickle.dump(storedata,f)
    print("data process sucess",storefilename)



    






