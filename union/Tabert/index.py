import json
import os
import pickle
import time
from typing import List

import numpy
import pandas as pd
import torch
from table_bert import Table, Column, TableBertModel
from torch import nn
from tqdm import tqdm
from transformers import *
import logging as log
import argparse


class Table:
    def __init__(self, id, header, data, context):
        self.id: str = id
        self.header: List[str] = header
        self.data: List[List[str]] = data
        self.context: str = context

    def __str__(self):  # for debug
        heading = "\t|||\t".join(self.header)
        body = ''
        for row in self.data: body += "\t|||\t".join(row) + '\n'
        return (f"______________________________________\n"
                f"Table ID:{self.id}\n"
                f"Context:{self.context}\n"
                f"{heading}\n"
                f"_____\n"
                f"{body}")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Qmodel = BertModel.from_pretrained(BERT_MODEL)
        self.Tmodel = TableBertModel.from_pretrained(
            TABERT_MODEL_PATH,
        )

    def forward(self, tp_table, tp_context, tn_table=None, tn_context=None):
        context_encoding, column_encoding, info = self.Tmodel.encode(
            contexts=[tp_context],
            tables=[tp_table]
        )
        return column_encoding


def build_dataset_predict(_tablePosList, _bertTokenizer, _tabertModel):
    tp_table_list = []
    tp_context_list = []
    for tp in tqdm(tablePosList):
        column_list = []
        head_list = tp.header
        value_1 = tp.data[0]
        print(type(value_1))
        for i in range(len(head_list)):
            column_list.append(Column(head_list[i].strip(), 'text', value_1[i]))

        table_p = Table(
            id=tp.id,
            header=column_list,
            data=tp.data
        ).tokenize(_tabertModel.tokenizer)
        tp_table_list.append(table_p)
        tp_context_list.append(_tabertModel.tokenizer.tokenize(tp.context))

    return list(zip(tp_table_list, tp_context_list))


def get_now() -> str:
    now = time.localtime()
    return "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)


def csv_to_list(csv_file, table_name) -> [List]:
    # 读取 CSV 文件并转换为 DataFrame
    table_pos_list = []
    data_source = pd.read_csv(csv_file, nrows=5, index_col=None)
    column_names = data_source.columns.tolist()
    data = data_source.values.tolist()
    num_rows, num_cols = data_source.values.shape[0], data_source.values.shape[1]
    jsonStr = {
        "id": table_name,
        "header": list(column_names),
        "data": data,
        "context": table_name,
        "num_cols": num_cols,
        "num_rows": num_rows
    }
    id = jsonStr['id']
    header = jsonStr['header']
    data = jsonStr['data']
    context = jsonStr['context']
    row = jsonStr['num_rows']
    col = jsonStr['num_cols']

    if col == 0:
        if DEBUG: print('col len error', col, id)
    elif row == 0:
        if DEBUG: print('row len error', row, id)
    table_pos_list.append(Table(id, header, data, context))
    return table_pos_list


def convert_all_csv_to_jsonl(folder_path, output_path):
    # 获取文件夹内所有CSV文件的路径
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    total_file_num = len(file_paths);
    current_index = 1;

    # 遍历每个 .csv 文件并进行转换，并将结果写入 .jsonl 文件
    with open(output_path, 'w') as jsonl_file:
        for csv_file in file_paths:
            table_name = os.path.splitext(os.path.basename(csv_file))[0]  # 提取表名，即去除后缀名的文件名
            print(table_name)
            json_data = csv_to_list(csv_file, table_name)
            json.dump(json_data, jsonl_file)
            jsonl_file.write('\n')  # 每个 JSON 数据占据一行
            print('当前进度' + str(current_index) + '/' + str(total_file_num))
            current_index = current_index + 1


def get_now_str() -> str:
    now = time.localtime()
    return "%04d_%02d_%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default='test')
    parser.add_argument("file_type", type='str', default='.csv')
    

    hp = parser.parse_args()
    dataFolder = hp.benchmark
    if 'large' in dataFolder:
        x = 'large'
    else:
        x = 'small'
    numpy.set_printoptions(suppress=True)
    TABERT_MODEL_PATH = hp.model_path
    BERT_MODEL = 'bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # >   Etc
    FILE_PATH = '/data_ssd/webtable/'+x+'/'+x+'_query/'
    FILE_TYPE = hp.file_type
    OUTPUT_PATH = "/data/" + dataFolder
    LOG_PATH = "/data/" + dataFolder
    DEBUG = False
    BATCH = 2
    # Config Area END_________________

    # BERT Model Load & Create model
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = Model()
    model.to(device)
    model.eval()  # 设置模型为评估模式
    LOG_FILE = LOG_PATH + get_now_str() + ".log"
    BATCH_SIZE = 3000
    # read csv file
    file_paths = [os.path.join(FILE_PATH, file) for file in os.listdir(FILE_PATH) if file.endswith(FILE_TYPE.lower())]
    total_file_num = len(file_paths)
    current_index = 1
    # 遍历每个 .csv 文件并进行转换，并将结果写入 .jsonl 文件
    index = 0
    w_index = 0
    with open(LOG_FILE, 'a') as logfile:
        dataEmbeds = []
        for csv_file in tqdm(file_paths):
            logfile.write(get_now() + ":start read csv_file.\n")
            # 提取表名，即去除后缀名的文件名
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            # 处理csv文件
            tablePosList = csv_to_list(csv_file, table_name)
            trainDataset = build_dataset_predict(tablePosList, tokenizer, model.Tmodel)
            # 写入.txt文件
            log('i', f"Total Train Data set Cnt : {len(trainDataset)}")
            # 将文件写入到指定文件夹
            with torch.no_grad():  # Disable gradient calculation for inference
                for tp_table, tp_context in trainDataset:
                    q_tp_embedding = model(tp_table, tp_context)
                    table_name = tp_table.id
                    dim0, dim1, _ = q_tp_embedding.shape
                    for i in range(dim0):
                        a_cpu = q_tp_embedding[i].cpu()
                        item = a_cpu.numpy()
                        dataEmbeds.append((table_name, item))
            print('当前进度' + str(current_index) + '/' + str(total_file_num))
            current_index = current_index + 1
            index = index + 1
            if (index % BATCH_SIZE == 0):
                data = []
                file_name = int(index / BATCH_SIZE)
                temp_output_path = OUTPUT_PATH + str(file_name) + '.pkl'
                data.append(dataEmbeds)
                pickle.dump(data, open(temp_output_path, "wb"))
                w_index = w_index + BATCH_SIZE
                print(w_index)
                print(len(dataEmbeds))
                dataEmbeds.clear()
                print(len(dataEmbeds))
                print('==========')
                logfile.write(get_now() + ":read csv_file end.\n")
        if (len(dataEmbeds) > 0):
            data = []
            file_name = int(index / BATCH_SIZE) + 1
            temp_output_path = OUTPUT_PATH + str(file_name) + '.pkl'
            data.append(dataEmbeds)
            pickle.dump(data, open(temp_output_path, "wb"))
            w_index = w_index + len(dataEmbeds)
            print(w_index)
        print('总处理数据len:')
        print(index)
        logfile.write("end.")
        logfile.close()
