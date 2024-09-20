import os
import pandas as pd
import numpy as np
import pickle
import random
import sys
import multiprocessing
from multiprocessing import Process, Queue
from sentence_transformers import  InputExample
from tqdm import  tqdm
import multiprocessing
import time
import nltk
import torch.multiprocessing
import csv

torch.multiprocessing.set_sharing_strategy('file_system')



def shuffle_sentence(sentence):
    # 将句子拆分为单词列表
    words = sentence.split()

    # 打乱单词列表
    random.shuffle(words)

    # 重新组合打乱后的单词列表为句子
    shuffled_sentence = ' '.join(words)

    return shuffled_sentence

def split_dataframe(df, num_parts):
    # 计算每份的行数
    num_rows = len(df)
    rows_per_part = num_rows // num_parts

    # 分割DataFrame
    parts = []
    for i in range(num_parts):
        start_index = i * rows_per_part
        end_index = (i + 1) * rows_per_part if i < num_parts - 1 else num_rows
        part_df = df.iloc[start_index:end_index]
        parts.append(part_df)

    return parts


def analyze_column_values(file_path, column_name):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 获取指定列的所有不同的列值数据和它们的频率
    value_counts = df[column_name].astype(str).value_counts()

    # 按照频率由高到低对列值进行排序
    sorted_values = value_counts.index.tolist()

    n = len(sorted_values)
    # 以逗号分隔列值
    col = ', '.join(sorted_values)

    # 统计列值的最大、最小和平均长度
    lengths = [len(str(value)) for value in sorted_values]
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    tokens = f"{column_name} contains {str(n)} values ({str(max_len)}, {str(min_len)}, {str(avg_len)}): {col}"
    # 返回结果

    tokens = nltk.word_tokenize(tokens)
    truncated_tokens = tokens[:512]
    truncated_sentence = ' '.join(truncated_tokens)

    return truncated_sentence



def process_task4(input_values,queue,queue_inforgather_train,queue_inforgather_evluate,file_train_path):
    # 执行任务，并返回结果列表
    train_samples = []
    dev_samples = []
    for index, row in input_values.iterrows():
        if row[0].strip() == "":
            continue
        if row[0].startswith("CAN"):
            file_train_path = "/data_ssd/opendata/small/datasets_CAN"
        elif row[0].startswith("SG"):
            file_train_path = "/data_ssd/opendata/small/datasets_SG"
        elif row[0].startswith("UK"):
            file_train_path = "/data_ssd/opendata/small/datasets_UK"
        elif row[0].startswith("USA"):
            file_train_path = "/data_ssd/opendata/small/datasets_USA"
        else:
            print("nofind this file",row[0])
        file1_path = os.path.join(file_train_path,row[0])

        if row[1].strip() == "":
            continue
        if row[1].startswith("CAN"):
            file_train_path = "/data_ssd/opendata/small/datasets_CAN"
        elif row[1].startswith("SG"):
            file_train_path = "/data_ssd/opendata/small/datasets_SG"
        elif row[1].startswith("UK"):
            file_train_path = "/data_ssd/opendata/small/datasets_UK"
        elif row[1].startswith("USA"):
            file_train_path = "/data_ssd/opendata/small/datasets_USA"
        else:
            print("nofind this file",row[1])

        file2_path = os.path.join(file_train_path,row[1])
        try:
            sentence_text1 = analyze_column_values(file1_path,row[2])
            sentence_text2 = analyze_column_values(file2_path, row[3])
        except Exception as e:
            continue
        score = float(row[4])  # Normalize score to range 0 ... 1
        random_number = random.random()
        flag = False
        if random_number < 0.2:
            flag= True

        if random_number > 0.8:
            dev_samples.append(sentence_text1 + "#####" + sentence_text2 + "#####" +  str(score))
        else:
            train_samples.append(sentence_text1 + "#####" + sentence_text2)

        if flag:
            shuffle_sentence1 = shuffle_sentence(sentence_text1)
            shuffle_sentence2 = shuffle_sentence(sentence_text2)
            random_number = random.random()
            if random_number > 0.8:
                dev_samples.append(shuffle_sentence1 + '#####' + shuffle_sentence2 + '#####' + str(score))
            else:
                train_samples.append(shuffle_sentence1 + "#####" + shuffle_sentence2)
        queue.put(1)
    queue_inforgather_train.put(train_samples)
    queue_inforgather_evluate.put(dev_samples)
    queue.put((-1, "test-pid"))

def train_process_single_ele(x):
    x_list = x.split("#####")
    s1= str(x_list[0])
    s2 = str(x_list[1])
    r = InputExample(texts=[s1, s2])
    return r

def dev_process_single_ele(x):
    try:
        x_list = x.split("#####")
        s1= str(x_list[0])
        s2 = str(x_list[1])
        score = float(x_list[2])
    except Exception as e:
        return ""
    
    r = InputExample(texts=[s1, s2], label=score)
    return r


def processdate(train_file,dev_file,num_partitions):
    train_input_list = []
    dev_input_list = []
    with open(train_file,"rb") as t_file:
        train_string_list = pickle.load(t_file)
    with open(dev_file,"rb") as d_file:
        dev_string_list= pickle.load(d_file)

    train_input_list = map(train_process_single_ele,train_string_list)
    dev_input_list = map(dev_process_single_ele,dev_string_list)

    train_input_list =  [i for i in train_input_list if i !=""]
    dev_input_list = [j for j in dev_input_list if j !=""]

    train_len = len(train_input_list) // num_partitions
    return_train = train_input_list[0:train_len]

    dev_len = len(dev_input_list) // num_partitions
    return_dev = dev_input_list[0:dev_len]

    return return_train,return_dev


def process_before_train(file_train_path,filecsv,filepath = "/data/lijiajun/deepjoin/pretrain_data_list/",name = "train_list.pkl",name2 = "evluate_list.pkl"):

    #file = '../sato_webtable_new.csv'
    #file = '../test.csv'
    #file_train_path = "/data_ssd/webtable/large/split_1"

    df = pd.read_csv(filecsv)
    inputs = df

    split_num = 10
    # 指定包含查询表的文件夹路径

    # 获取文件夹中的所有文件名
    
    sub_file_ls = split_dataframe(inputs, split_num)
    process_list = []

    #####
    # 为每个进程创建一个队列
    queues = [Queue() for i in range(split_num)]
    # queue = Queue()
    # 一个用于标识所有进程已结束的数组
    finished = [False for i in range(split_num)]

    # 为每个进程创建一个进度条
    bars = [tqdm(total=len(sub_file_ls[i]), desc=f"bar-{i}", position=i) for i in range(split_num)]
    # 用于保存每个进程的返回结果
    results = [None for i in range(split_num)]
    queue_inforgather_train = multiprocessing.Manager().Queue()
    queue_inforgather_evluate = multiprocessing.Manager().Queue()

    for i in range(split_num):
        process = Process(target=process_task4, args=(sub_file_ls[i], queues[i], queue_inforgather_train,queue_inforgather_evluate,file_train_path))
        process_list.append(process)
        process.start()

    while True:
        for i in range(split_num):
            queue = queues[i]
            bar = bars[i]
            try:
                # 从队列中获取数据
                # 这里需要用非阻塞的get_nowait或get(True)
                # 如果用get()，当某个进程在某一次处理的时候花费较长时间的话，会把后面的进程的进度条阻塞着
                # 一定要try捕捉错误，get_nowait读不到数据时会跑出错误
                res = queue.get_nowait()
                if isinstance(res, tuple) and res[0] == -1:
                    # 某个进程已经处理完毕
                    finished[i] = True
                    results[i] = res[1]
                    continue
                bar.update(res)
            except Exception as e:
                continue

                # 所有进程处理完毕
        if all(finished):
            break

    for process in process_list:
        process.join()

    train_list = []
    evaluate_list = []

    while not queue_inforgather_train.empty():
        try:
            k = queue_inforgather_train.get_nowait()
            train_list += k
            del k
        except Exception as e:
            continue

    while not queue_inforgather_evluate.empty():
        try:
            k = queue_inforgather_evluate.get_nowait()
            evaluate_list += k
            del k
        except Exception as e:
            continue


##-----------------------------------这里需要对队列中的数据进行存储-------------------------
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath,name),'wb') as file:
        pickle.dump(train_list,file)

    with open(os.path.join(filepath,name2),'wb') as file:
        pickle.dump(evaluate_list,file)
    print("pretrain_data pickle sucesss")

def transform_train_dev_toInput(filepath="/data/lijiajun/deepjoin/pretrain_data_list/",name="train_list.pkl",name2="evluate_list.pkl",splitnumn=1):
##--------------------------------将存储的pkl文件 转换成数据inputexample数据 --------------------------------------
    train_file = os.path.join(filepath,name)
    dev_file = os.path.join(filepath,name2)
    train,dev = processdate(train_file,dev_file,splitnumn)
    return train,dev


    












