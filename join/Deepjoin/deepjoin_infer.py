import argparse

from torch.utils.data import DataLoader
import math
import torch
from deepjoin.deepjoinitem.Forward1 import process_onedataset
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.nn.parallel import DataParallel
import logging
from datetime import datetime
import os
import csv
from dataprocess.multi_preocess_csv import process_before_train, transform_train_dev_toInput

from train import train
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset")
parser.add_argument("--datafile", help="datafile")
parser.add_argument("--storepath", help="storepath")

args = parser.parse_args()

dataset  = args.dataset_para
datafile = args.datafile
storepath= args.storepath
if __name__ == '__main__':
    if  dataset == "opendata":
        # filedir = "/data/lijiajun/deepjoin_webtables/large"
        # filename = "deepjoin_split1.pkl"
        # datafile = os.path.join(filedir,filename)
        # datafile = datafile
        # storepath = "/data/final_result/deepjoin/webtable/"
        process_onedataset(datafile,storepath = storepath)
    else:
        process_onedataset(datafile,storepath = storepath)

