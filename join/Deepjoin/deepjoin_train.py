import argparse

from torch.utils.data import DataLoader
import math
import torch
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
parser.add_argument("--model_name", help="model_name")
parser.add_argument("--model_save_path", help="model_save_path")
parser.add_argument("--file_train_path", help="file_train_path")
parser.add_argument("--tain_csv_file", help="tain_csv_file")
parser.add_argument("--storepath", help="storepath")
args = parser.parse_args()

# 获取特定的参数
dataset_para = args.dataset
model_name = args.model_name
model_save_path = args.model_save_path
file_train_path = args.file_train_path
tain_csv_file = args.tain_csv_file
storepath = args.storepath

if dataset_para == "opendata":

    abspath_cav = os.path.abspath(tain_csv_file)
    process_before_train(file_train_path,abspath_cav,
                         filepath = storepath,
                         name = "train_opendata_list.pkl",name2 = "evluate_opendata_list.pkl")

    #opendata train size ：129823 test_siez: 32443
    train_samples,dev_samples = transform_train_dev_toInput(storepath, name = "train_opendata_list.pkl",
                                                            name2 = "evluate_opendata_list.pkl",splitnumn=100)

    train(train_samples,dev_samples,model_save_path)
else:
    abspath_cav = os.path.abspath(tain_csv_file)
    process_before_train(file_train_path,abspath_cav)
    train_samples,dev_samples = transform_train_dev_toInput(splitnumn=1)

    train(train_samples,dev_samples,model_save_path,cpuid = 0)