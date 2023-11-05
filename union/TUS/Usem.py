import os
import ast
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool#多进程库
from functools import partial
from datasketch import MinHash, MinHashLSHForest
from concurrent.futures import ThreadPoolExecutor#多线程库
from collections import Counter
from yago import get_yagoTypes_from_tsv, process_string

def fill_zero(data):
    #data: 是一个字典，key是列名，values是对应的每列的列表
    maxlen = 0
    for key in data.keys():
        maxlen = max(maxlen, len(data[key]))
    
    for key in data.keys():
        if len(data[key]) < maxlen:
            dif = maxlen - len(data[key])
            data[key] += [0] * dif

    return data
#三个分布搞个新的，三个LSH生成新的，

def save_list_to_file(file_path, my_list):
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')

def read_list_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
        return [ast.literal_eval(line.strip()) for line in content]


def match_entity_from_yago():
    
    #创建虚拟表，用以执行全文搜索
    conn, cursor = get_yagoTypes_from_tsv()

    #先一把所有表的所有行都映射到yago中的类，使用match，全文搜索
    floder_path = 'benchmark'
    save_floder_path = 'benchmarSem'
    for file_name in os.listdir(floder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(floder_path, file_name)
            df = pd.read_csv(file_path)
            saved_data = {}
            #取出每一列，将列的每个值都先映射到实体，选出最多三个实体，把这个实体映射到类中
            for column in df.columns:
                values = df[column].to_list()
                values = [s for s in values if s!=" "]
                #把所有的值都处理成小写，并用空格来代替标点符号
                for i in range(len(values)):
                    values[i] = process_string(values[i])
                counter = Counter(values)
                #统计值出现的次数，形成一个字典
                values_set= set(values)
                classes_of_thisColumn = []
                for value in values_set:
                    search_keywords = value
                    query = f"select distinct entity from yagoTypes where entity match '{search_keywords}' order by bm25(yagoTypes) desc limit 3;"
                    
                    cursor.execute(query)
                    results = cursor.fetchall()
                    #找到了这个value匹配的实体名称，还要依次把相应的taxonomy_class类找出来
                    for result in results:
                        query = f'''SELECT distinct taxonomy_class
                        FROM yagoTypes
                        JOIN yagoTaxonomy
                        ON yagotypes.type = yagoTaxonomy.subclass
                        WHERE yagoTypes.entity = '{result[0]}';
                        '''
                        cursor.execute(query)
                        temp = cursor.fetchall()
                        for i in range(len(temp)):
                            temp[i] = temp[i][0]

                        #根据列中value出现的次数，添加相应个数的class值
                        classes_of_thisColumn += temp * counter[value]
                        #查询返回的是一个元组，需要把元组中的[0]拿出来，然后放进classes_of_thisColumn, 并且需要distinct
                
                #把这个列的属于的所有类都存到字典中
                saved_data[column] = classes_of_thisColumn
            
            #创建DataFrame对象，把saved_data字典加载到这里面，最后使用to_csv将这个表对应的sem表保存
            saved_df = pd.DataFrame(saved_data)
            saved_file_path = os.path.join(save_floder_path, file_name)
            saved_df.to_csv(saved_file_path, index=False)
            print("sucess saved on " + saved_file_path) 

    cursor.close()
    conn.close()


def query_lsh(lsh, value, k=3):
    minHash_query = MinHash(num_perm=128)
    minHash_query.update(value.encode('utf-8'))

    results = lsh.query(minHash_query, k)
    
    return results


def match_yago_entity_from_yago_lsh(lsh):
    floder_path = '/data_ssd/webtable/large/split_1'
    save_floder_path = '/data_ssd/tus/webtable/usem/split_1'
    conn, cursor = get_yagoTypes_from_tsv()
    exist_list = os.listdir(save_floder_path)
    floder_list = os.listdir(floder_path)
    cal_list = [x for x in floder_list if x not in exist_list] 
    for file_name in cal_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(floder_path, file_name)
            df = pd.read_csv(file_path)
            saved_data = {}

            for column in df.columns:
                values = df[column].to_list()
                values = [process_string(s) for s in values if s!=' ' and s!='']
                counter = Counter(values)
                values_set = set(values)

                classes_of_thisColumn = []
                for value in values_set:
                    search_keywords = value
                    results = query_lsh(lsh, search_keywords, 3)

                    for result in results:
                        query = f'''SELECT distinct taxonomy_class
                        FROM yagoTypejoinclasses
                        WHERE entity = '{result}';
                        '''
                        cursor.execute(query)
                        temp = cursor.fetchall()
                        for i in range(len(temp)):
                            temp[i] = temp[i][0]

                        classes_of_thisColumn += temp * counter[value]
                saved_data[column] = classes_of_thisColumn

            saved_data = fill_zero(saved_data)
            saved_df = pd.DataFrame(saved_data)
            saved_file_path = os.path.join(save_floder_path, file_name)
            saved_df.to_csv(saved_file_path, index=False)
            print('success saved on' + saved_file_path) 

    conn.commit()
    cursor.close()
    conn.close()


def match_yago_entity_from_yago_lsh_thread(lsh):
    floder_path = 'benchmark'
    save_floder_path = 'UsemLshTest'
    exist_list = os.listdir(save_floder_path)
    floder_list = os.listdir(floder_path)
    cal_list = [x for x in floder_list if x not in exist_list] 

    def process_subset(file_list):
        conn, cursor = get_yagoTypes_from_tsv()
        for file_name in file_list:
            if file_name.endswith('.csv') and file_name not in exist_list:
                file_path = os.path.join(floder_path, file_name)
                df = pd.read_csv(file_path)
                saved_data = {}

                for column in df.columns:
                    values = df[column].to_list()
                    values = [process_string(s) for s in values if s != ' ' and s != '']
                    counter = Counter(values)
                    values_set = set(values)

                    classes_of_thisColumn = []
                    for value in values_set:
                        search_keywords = value
                        results = query_lsh(lsh, search_keywords, 1)

                        for result in results:
                            query = f'''SELECT distinct taxonomy_class
                            FROM yagoTypejoinclasses
                            WHERE entity = '{result}';
                            '''
                            cursor.execute(query)
                            temp = cursor.fetchall()
                            for i in range(len(temp)):
                                temp[i] = temp[i][0]

                            classes_of_thisColumn += temp * counter[value]
                    saved_data[column] = classes_of_thisColumn

                saved_data = fill_zero(saved_data)
                saved_df = pd.DataFrame(saved_data)
                saved_file_path = os.path.join(save_floder_path, file_name)
                saved_df.to_csv(saved_file_path, index=False)
                conn.commit()
                cursor.close()
                conn.close()
                print('success saved on' + saved_file_path)
    cal_sublists = np.array_split(cal_list, 16)
    cal_sublists = [c.tolist() for c in cal_sublists]

    with ThreadPoolExecutor(max_workers=16) as executor:
        # Create threads for each sublist
        for sublist in cal_sublists:
            executor.submit(process_subset, sublist)


def yago_entity_lsh(folder_path='yago', file_name='yagoTypes.tsv', threshold = 0.7):
    file_path = os.path.join(folder_path, file_name)
    forest = MinHashLSHForest(num_perm=128)
    entity_set = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            _, entity, relation, type = line.strip().split('\t') 
            entity = process_string(entity[1:-1])
            entity_set.add(entity)

    idx = 0
    print(len(entity_set))
    for entity in entity_set:
        minhash = MinHash(num_perm=128)
        minhash.update(entity.encode('utf-8'))
        forest.add(entity, minhash)
        print(idx)
        idx+=1
    #构建索引
    forest.index()

    return forest


#先获得临时的可以使用的几个文件，保存到一个临时文件夹之中，然后写Usem lsh以及Usem的具体值，这里

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="命令行参数示例")
    # parser.add_argument('-n', '--num', type=int, default=0, help="这是一个可选参数，用于传入值")
    # args = parser.parse_args()
    #match_entity_from_yago()
    # lsh = yago_entity_lsh()
    # with open("UsemLSH.pkl", "wb") as f:
    #     pickle.dump(lsh, f)
    # cal_list = os.listdir('benchmark')
    # exist_list = os.listdir('UsemLshTest')
    # cal_list = [x for x in cal_list if x not in exist_list]
    # cal_sublists = np.array_split(cal_list, 6)
    # cal_sublists = [c.tolist() for c in cal_sublists]

    start_time = time.time()
    with open("lsh/yagoLSH.pkl", "rb") as f:
        lsh = pickle.load(f)
    end_time = time.time()
    print("load LSH use" + str(end_time-start_time)+'s')
    #file_path = 'finalList.txt'
    # 从文件中读取列表
    #read_list = read_list_from_file(file_path)

    match_yago_entity_from_yago_lsh(lsh)