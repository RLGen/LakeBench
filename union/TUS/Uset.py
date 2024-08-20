import os
import json
import random
import pickle
import sys
import argparse
import multiprocessing 
import itertools
import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH, MinHashLSHForest
from scipy.stats import hypergeom
from collections import Counter
from Unl import u_nl, save_dict_to_json


def majority_classes(classes):
    #返回多数类，返回结果是一个列表，如果有多个类出现的次数相同那就都返回
    classes = [x for x in classes if x != 0]
    #如果是空列表，那就返回空的
    if len(classes) == 0:
        return []

    counter = Counter(classes)

    # 获取出现次数最多的key
    max_count = max(counter.values())
    most_common_keys = [key for key, count in counter.items() if count == max_count]

    return most_common_keys


#计算两个列直接的u_set分数
def u_set_or_sem(type, a, b):
    if type == 'Usem':
        #这里的set_a是一个列表，但是因为majority函数返回的是字典的key，所以不会有重复值，就用一样的名字来命名了
        # set_a = set(majority_classes(a))
        # set_b = set(majority_classes(b))
        if a[-1] != 0:
            set_a = set([a[-1]])
        else:
            set_a = set([])

        if b[-1] != 0:
            set_b = set([b[-1]])
        else:
            set_b = set([]) 
    else:
        set_a = set(a)
        set_b = set(b)

    n_a = len(set_a)
    n_b = len(set_b)
    n_d = n_a + n_b
    
    intersection = set_a & set_b
    intersection_list = list(intersection)
    t = len(intersection_list)
    
    #计算超几何分布
    hypergeom_dist = hypergeom(n_d, n_a, n_b)
    
    #计算累积分布
    cdf = hypergeom_dist.cdf(t)
    #print(hypergeom_dist.pmf(t))
    #print(cdf)
    return cdf


#所有的表的所有列根据Jaccard相似性建立LSH index 
def minhash_Lsh_forest(type='Uset', folder_path = 'benchmark', t = 0.7, n = 128):

    forest = MinHashLSHForest(num_perm=n)

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            #取出每一列，将每一列的值minhah，然后加入lsh中，创建索引，列的索引用文件名+列名
            for column in df.columns:
                values = df[column].to_list()
                if type=='Usem':
                    #执行选多数的操作
                    values_set = majority_classes(values)
                elif type=='Uset':
                    #去重操作
                    values_set = set(values)
                
                #如果这一列找不到Yago里的entity，那就跳过
                if len(values_set) == 0:
                    continue

                minHash = MinHash(num_perm=n)
                for element in values_set:
                    element = str(element)
                    minHash.update(element.encode('utf-8'))
                forest.add(filename+" "+column, minHash)
    forest.index()

    return forest


def split_list(lst, num_parts):
    avg = len(lst) // num_parts
    remainder = len(lst) % num_parts

    result = []
    start = 0
    for i in range(num_parts):
        if i < remainder:
            end = start + avg + 1
        else:
            end = start + avg
        result.append(lst[start:end])
        start = end

    return result

def minhash_multi_process(folder_path, file_list, q, spl, id, n, type, t):
    # print("{} start".format(id))
    lsh = MinHashLSH(threshold=t, num_perm=128)

    for i, filename in enumerate(file_list):
        if filename.endswith('.csv'):
            #file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(filename, lineterminator='\n', low_memory=False)
            #取出每一列，将每一列的值minhah，然后加入lsh中，创建索引，列的索引用文件名+列名
            for column in df.columns:
                values = df[column].to_list()
                if type=='Usem':
                    #执行选多数的操作
                    #values_set = majority_classes(values)
                    if values[-1] == 0:
                        continue
                    values_set = [values[-1]]
                elif type=='Uset':
                    #去重操作
                    values_set = set(values)
                
                #如果这一列找不到Yago里的entity，那就跳过
                if len(values_set) == 0:
                    continue

                minHash = MinHash(num_perm=n)
                values_set = list(values_set)
                values_set = [str(value) for value in values_set]
                minHash.update_batch([value.encode('utf-8') for value in values_set])
                lsh.insert(filename+" "+column, minHash)
        if id==0:
            sys.stdout.write("\rId 0 Process Read and minhash {}/{} files".format(i+1, len(file_list)))
    if id==0:
        sys.stdout.write("\n")
    q.put(lsh)

def minhash_Lsh(type='Uset', folder_path = 'benchmark', t=0.9, n =128):
    lsh = MinHashLSH(threshold=t, num_perm=n)
    split_num = 40
    #with open('/data_ssd/tus/webtable/lsh/UsetLSH_split_2.pkl', "rb") as f:
        #lsh = pickle.load(f)
    #child_folders = ['split_'+str(i) for i in range(1,7)]
    #child_folders = ['datasets_CAN', 'datasets_SG', 'datasets_UK', 'datasets_USA']
    child_folders = ['datasets_CAN']
    file_list = []
    for child_folder in child_folders:
        files = os.listdir(folder_path+child_folder)
        files = [os.path.join(folder_path+child_folder, f) for f in files]
        file_list.extend(files)
    #file_list = os.listdir(folder_path)
    spl = folder_path.split('/')[-1]

    # 读入候选集目录
    sub_sets_files = split_list(file_list, split_num)
    process_list = []
    # pocessing
    entries_q = multiprocessing.Queue()
    for i in range(split_num):
        process = multiprocessing.Process(target=minhash_multi_process, args=(folder_path, sub_sets_files[i], entries_q, spl, i, n, type, t))
        process.daemon = True
        process_list.append(process)
        process.start()
    for i in range(split_num):
        lsh.union(entries_q.get())
        sys.stdout.write("\rRead and minhash {}/{} sets".format(i+1,split_num))
    sys.stdout.write("\n")
    for i, process in enumerate(process_list):
        process.join()

    # for filename in tqdm(file_list):
    #     if filename.endswith('.csv'):
    #         file_path = os.path.join(folder_path, filename)
    #         df = pd.read_csv(file_path)
    #         #取出每一列，将每一列的值minhah，然后加入lsh中，创建索引，列的索引用文件名+列名
    #         for column in df.columns:
    #             values = df[column].to_list()
    #             if type=='Usem':
    #                 #执行选多数的操作
    #                 #values_set = majority_classes(values)
    #                 if values[-1] == 0:
    #                     continue
    #                 values_set = [values[-1]]
    #             elif type=='Uset':
    #                 #去重操作
    #                 values_set = set(values)
                
    #             #如果这一列找不到Yago里的entity，那就跳过
    #             if len(values_set) == 0:
    #                 continue

    #             minHash = MinHash(num_perm=n)
    #             values_set = list(values_set)
    #             values_set = [str(value) for value in values_set]
    #             minHash.update_batch([value.encode('utf-8') for value in values_set])
    #             lsh.insert(spl + r'/' + filename+" "+column, minHash)
    
    return lsh


def query_lsh(values, lsh, type='Uset', n = 128, folder_path='benchmark'):
    if type=='Usem':
            #执行选多数的操作
        #values_set = majority_classes(values)
        if values[-1]==0:
            return []
        values_set = [values[-1]]
    elif type=='Uset':
        #去重操作
        values_set = set(values)
    
    #如果这一列找不到Yago里的entity，那就跳过
    if len(values_set) == 0:
        return []
    
    minHash_query = MinHash(num_perm=128)
    values = list(values_set)
    values = [str(v) for v in values]
    minHash_query.update_batch([v.encode('utf-8') for v in values])
    
    results = lsh.query(minHash_query)
    
    return results


#根据minhash lsh的分桶结果来计算u_set分数的分布
def u_set_or_sem_distribution(lsh, type = 'Uset', folder_path = 'benchmark'):
    
    u_set_or_sem_all = []
    hashtables = lsh.hashtables

    print(len(hashtables))
    sampling_ratio = 0.05
    table_id = 0
    for table in hashtables:
        idx = 0
        print('table:'+str(table_id)+'has '+str(len(table.keys()))+' buckets.')
        table_id += 1
        for bucket in table.keys():
            #取出一个桶中所有的值
            values_in_bucket = table[bucket]
            print('bucket:'+str(idx))
            idx += 1
            #如果桶中的值少于两个就跳过这个桶
            if len(values_in_bucket)<2:
                continue

            #把所有的两两组和枚举出来
            np.random.seed(41)
            combinations = list(itertools.combinations(values_in_bucket, 2))
            length = len(combinations)
            if length<500:
                size = length
            else:
                size = max(1, int(len(combinations)*sampling_ratio))
            size = min(size, 2000)
            random_numbers = np.random.randint(low=0, high=len(combinations), size=size)
            subset = [combinations[i] for i in random_numbers]
            #pair_id=0
            for pair in tqdm(subset):
                file_name1, column_name1 = pair[0].split(maxsplit=1)#只按照第一个空格分
                file_name2, column_name2 = pair[1].split(maxsplit=1)
                column1 = pd.read_csv(folder_path + '/' + file_name1)[column_name1].to_list()
                column2 = pd.read_csv(folder_path + '/' + file_name2)[column_name2].to_list()
                if type=='Usem' and (column1[-1]==0 and column2[-1] == 0):
                    continue
                u_set_or_sem_all.append(u_set_or_sem(type, column1, column2))
                #pair_id += 1
                #print(pair_id)
    # #统计每个值出现的次数
    # counts = Counter(u_set_or_sem_all)

    # #统计所有的数出现的总次数
    # count = sum(list(counts.values()))

    # #更新字典中的次数为概率
    # for key in counts:
    #     counts[key] = counts[key]/count

    return u_set_or_sem_all


#读取保存的分布
def read_distribution(json_file):
    with open(json_file, 'r') as file:
        distribution = json.load(file)

        return distribution


def query_lsh_forest(values, lsh, type='Uset', threshold=0.7, n = 128, folder_path='benchmark', k=10**10):
    #对表中的一列即values执行lsh query
    values_set = set(values)
    minHash_query = MinHash(num_perm=128)
    for element in values_set:
        element = str(element)
        minHash_query.update(element.encode('utf-8'))
    
    candidates = lsh.query(minHash_query, k)
    results = []
    
    #计算精确的jaccard相似度，返回大于等于阈值的结果
    for candidate in candidates:
        file_name, column_name = candidate.split(maxsplit=1)
        file_path = os.path.join(folder_path, file_name)
        df_query = pd.read_csv(file_path)
        values = df_query[column_name].to_list()
        if type=='Usem':
            #执行选多数的操作
            values_set = majority_classes(values)
        elif type=='Uset':
            #去重操作
            values_set = set(values)
        
        #如果这一列找不到Yago里的entity，那就跳过
        if len(values_set) == 0:
            continue

        candidate_minhash = MinHash(num_perm=n)
        for element in values_set:
            element = str(element)
            candidate_minhash.update(element.encode('utf-8'))

        similarity  = minHash_query.jaccard(candidate_minhash)
        if similarity >= threshold:
            results.append(candidate)

    return results


def cal_precise_unionablity(type, values_of_this_column, lsh_result, folder_path, model):
    #计算values_of_this_column与候选列的精确可并行分数
    #后面加上u_NL的分数
    col_dict = {}
    for r in lsh_result:
        file_name, column_name = r.split(maxsplit=1)
        file_path = os.path.join(folder_path, file_name)
        df_query = pd.read_csv(file_path)
        if type == 'Uset' or type == 'Usem':
            unionablity = u_set_or_sem(type, values_of_this_column, df_query[column_name].to_list())
        else:
            unionablity = u_nl(values_of_this_column, df_query[column_name], model)
        if file_name in col_dict:
            if unionablity > col_dict[file_name]:
                col_dict[file_name] = unionablity
        else:
            col_dict[file_name] = unionablity
    
    return col_dict


#save_dict_to_json(u_set_distribution(u_set_Lsh()), 'result/uset_lsh.json')
if __name__ == "__main__":
    # u_sem_lsh = minhash_Lsh('Usem','benchmarkSemTemp')
    # unique_values = os.listdir('benchmarkSemTemp')
    # np.random.seed(41)
    # random_values = np.random.choice(unique_values, size=1, replace=False)
    # query_tables = random_values.tolist()

    # df = pd.read_csv('benchmarkSemTemp/' + query_tables[0])
    # for column in df.columns:
    #     result = query_lsh(df[column], u_sem_lsh, type='Usem', folder_path='benchmarkSemTemp')
    #     print(result)
    #选择一些表，查询与它相关的列
    # with open("lsh/UsetLSH.pkl", "rb") as f:
    #     u_set_lsh = pickle.load(f)
    # save_dict_to_json(u_set_or_sem_distribution(u_set_lsh, 'Uset', 'benchmark'), 'distribution/uset_lsh_new.json')
    # with open('lsh/UsemLSH.pkl', 'rb') as f:
    #     u_sem_lsh = pickle.load(f)
    # save_dict_to_json(u_set_or_sem_distribution(u_sem_lsh, 'Usem', 'UsemLshTest'), 'distribution/usem_lsh_7.30.json')
    # uset_lsh = minhash_Lsh('Uset', folder_path='benchmark')
    # unique_values = os.listdir('benchmark')
    # np.random.seed(41)
    # random_values = np.random.choice(unique_values, size=1, replace=False)
    # query_tables = random_values.tolist()

    # df = pd.read_csv('benchmark/' + query_tables[0])
    # for column in df.columns:
    #     result = query_lsh(df[column].to_list(), uset_lsh, type='Uset', folder_path='benchmark')
    #     print(result)
    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument('-s', '--start', type=int, default=1, help="start folder_path number")
    args = parser.parse_args()
    lsh = minhash_Lsh(type='Uset', folder_path='/data_ssd/opendata/small/')
    #lsh = minhash_Lsh(type='Uset', folder_path='/data_ssd/opendata/small/datasets_SG')
    with open('/data_ssd/tus/opendata/lsh/UsetLSH_small.pkl', 'wb') as f:
        pickle.dump(lsh, f)
