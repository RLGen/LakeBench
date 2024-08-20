import os
import csv
import glob
import time
import argparse
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import os.path
import generalFunctions as genFunc

from tqdm import tqdm
from Usem import fill_zero

def computeColumnSemantics(input_table, LABEL_DICT, TYPE_DICT, CLASS_DICT):
    col_id = 0
    not_found_in_yago = []
    column_dictionary = {}
    for columnName in input_table.columns:
        if genFunc.getColumnType(input_table[columnName].tolist()) == 1: #check column Type
            input_table[columnName] = input_table[columnName].map(str)      
            #get unique values in the column and preprocess them.
            value_list = genFunc.preprocessListValues(input_table[columnName].unique())
            #search values in KB
            all_found_types = {}
            total_kb_hits = 0
            for value in value_list:
                current_entities = set()
                current_types = set()
                 
                current_entities = LABEL_DICT.get(value, "None")
                if current_entities != "None": #found in KB
                    total_kb_hits += 1
                    for entity in current_entities:
                        if entity in TYPE_DICT:
                            temp_type = TYPE_DICT[entity]
                            for entity_type in temp_type:
                                current_types.add(entity_type)
                    for each_type in current_types:
                        if each_type in all_found_types:
                            all_found_types[each_type] +=1
                        else:
                            all_found_types[each_type] = 1
                else:
                    not_found_in_yago.append(value)    
                
            #find the top-level type with highest count.
            all_top_types = [v for v in sorted(all_found_types.items(), key=lambda kv: (-kv[1], kv[0])) if v[0] in CLASS_DICT]
            if all_top_types:
              selected_top_type = all_top_types[0][0]
              top_type_count = all_top_types[0][1]             
              children_of_top_types = CLASS_DICT[selected_top_type]
                #add children of top types to the bag of word
              for each in all_found_types:
                if each in children_of_top_types and (all_found_types[each] / top_type_count) >= 0:
                    if columnName not in column_dictionary:
                        column_dictionary[columnName] = [each]
                    else:
                        column_dictionary[columnName].append(each)
        col_id += 1

    return column_dictionary#处理一下，改成一个DataFrame


 # 定义处理单个表格的函数
def process_table(table, label_dict, type_dict, class_dict, save_file_path):
    table_name = table.rsplit(os.sep, 1)[-1]
    input_table = pd.read_csv(table, lineterminator='\n', low_memory=False)
    column_semantic = computeColumnSemantics(input_table, label_dict, type_dict, class_dict)
    if len(column_semantic) == 0:
        return
    saved_data = fill_zero(column_semantic)
    saved_df = pd.DataFrame(saved_data)
    saved_file_path = save_file_path + table_name
    saved_df.to_csv(saved_file_path, index=False)
    #print("正在处理表格：", table_name)


# 定义处理所有表格的主函数
def process_tables(query_table_file_name, label_dict, type_dict, class_dict, save_file_path):
    pool = multiprocessing.Pool(processes=20)
    func = partial(process_table, label_dict=label_dict, type_dict=type_dict, class_dict=class_dict, save_file_path=save_file_path)
    pool.map(func, query_table_file_name)
    pool.close()
    pool.join()


def parrel_main():
    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument('-s', '--start', type=int, default=1, help="start folder_path number")
    args = parser.parse_args()
    child_folders = ['datasets_CAN/', 'datasets_SG/', 'datasets_UK/', 'datasets_USA/']
    YAGO_PATH = r"/home/yuyanrui/santos/yago/yago_pickle/"

    LABEL_FILE_PATH = YAGO_PATH + "yago-wd-labels_dict.pickle"
    TYPE_FILE_PATH = YAGO_PATH + "yago-wd-full-types_dict.pickle" 
    CLASS_FILE_PATH = YAGO_PATH + "yago-wd-class_dict.pickle"
    
    # SAVE_FILE_PATH = r"/data_ssd/tus/webtable/santos_usem/split_" + str(args.start) + '/'
    SAVE_FILE_PATH = r"/data_ssd/tus/opendata/santos_usem/" + child_folders[args.start]

    label_dict = genFunc.loadDictionaryFromPickleFile(LABEL_FILE_PATH)
    type_dict = genFunc.loadDictionaryFromPickleFile(TYPE_FILE_PATH)
    class_dict = genFunc.loadDictionaryFromPickleFile(CLASS_FILE_PATH)

    #query_table_file_name = glob.glob(r'/data_ssd/webtable/large/split_'+str(args.start)+"/*.csv")
    query_table_file_name = glob.glob(r'/data_ssd/opendata/small/'+ child_folders[args.start] + "*.csv")
    # 调用主函数处理所有表格
    process_tables(query_table_file_name, label_dict, type_dict, class_dict, SAVE_FILE_PATH)


def main():

    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument('-s', '--start', type=int, default=1, help="start folder_path number")
    args = parser.parse_args()
    YAGO_PATH = r"/home/yuyanrui/santos/yago/yago_pickle/"

    LABEL_FILE_PATH = YAGO_PATH + "yago-wd-labels_dict.pickle"
    TYPE_FILE_PATH = YAGO_PATH + "yago-wd-full-types_dict.pickle" 
    CLASS_FILE_PATH = YAGO_PATH + "yago-wd-class_dict.pickle"
    
    SAVE_FILE_PATH = r"/data_ssd/tus/webtable/santos_usem/split_/" + str(args.start)

    label_dict = genFunc.loadDictionaryFromPickleFile(LABEL_FILE_PATH)
    type_dict = genFunc.loadDictionaryFromPickleFile(TYPE_FILE_PATH)
    class_dict = genFunc.loadDictionaryFromPickleFile(CLASS_FILE_PATH)

    query_table_file_name = glob.glob(r'/data_ssd/webtable/large/split_/'+str(args.start)+"*.csv")

    for table in tqdm(query_table_file_name):
        table_name = table.rsplit(os.sep, 1)[-1]
        input_table = pd.read_csv(table)
        column_sematic = computeColumnSemantics(input_table, label_dict, type_dict, class_dict)
        if len(column_sematic) == 0:
            continue
        
        saved_data = fill_zero(column_sematic)
        saved_df = pd.DataFrame(saved_data)
        table_name = table.rsplit(os.sep, 1)[-1]
        saved_file_path = SAVE_FILE_PATH + table_name
        saved_df.to_csv(saved_file_path, index=False)
        #print("Processing Table:", table_name)


if __name__ == "__main__":
    parrel_main()