import os
import time
import json
import random
import pickle
import itertools
import concurrent.futures
import numpy as np
import pandas as pd
import scipy.stats as stats

from tqdm import tqdm
from collections import Counter
from gensim.models import KeyedVectors
from simhash import Simhash, SimhashIndex
from concurrent.futures import ProcessPoolExecutor
from yago import process_string
#from Uset import save_dict_to_json


def save_dict_to_json(values_list, file_path):
    num_bins = 500
    histogram = [0] * num_bins
    for value in values_list:
        bin_index = max(0, min(int(value * num_bins), num_bins - 1))
        histogram[bin_index] += 1

    total_values = len(values_list)
    frequency_dict = {}
    for i in range(num_bins):
        right_value = (i + 1) / num_bins
        frequency = round(histogram[i] / total_values, 6)
        frequency_dict[right_value] = frequency
    
    #确保所有频率加起来是1
    total_frequency = sum(frequency_dict.values())
    for key in frequency_dict:
        frequency_dict[key] /= total_frequency

    with open(file_path, 'w') as file:
        json.dump(frequency_dict, file)

    print("data saved as json in:"+ file_path)


def cumulative_frequency(file_path, given_value):
    cumulative_freq = 0.0
    with open(file_path, 'r') as file:
        frequency_dict = json.load(file)
    
    for right_value, frequency in frequency_dict.items():
        if float(right_value) <= given_value:
            cumulative_freq += frequency

    return cumulative_freq


def read_goodness_from_dict(file_path, value):
    with open(file_path, 'r') as f:
        frequency_dict = json.load(f)
    
    num_bins = len(frequency_dict)
    bin_index = int(value * num_bins)
    right_value = (bin_index + 1) / num_bins
    frequency = frequency_dict.get(right_value, 0.0)
    
    return frequency

#计算两列之间的u_nl
def u_nl(a, b, model):
    #a,b进来之后得先读取向量， 然后才能算，或者这里传进来一个fastText的model
    def find_word2vec_of_one_column(values, model):
        #查找这一列的fastText embedding，并返回均值和协方差
        values = [value for value in values if value!=" " and value!=""]
        values = [process_string(value) for value in values]
        counter = Counter(values)
        
        values_set = set(values)
        embedding_of_thisColumn = []

        for value in values_set:
            tokens = value.split()
            embeddings = []
            #把每一个token分别查询，最后的embedding是所有token的平均值
            for token in tokens:
                if token in model.key_to_index:
                    embeddings.append(model[token])
            if len(embeddings) == 0:
                continue
            
            array = np.array(embeddings)
            average_vector = np.mean(array, axis=0)
            
            for _ in range(counter[value]):
                embedding_of_thisColumn.append(average_vector)
        length = len(embedding_of_thisColumn)
        if length > 1:
            array_of_thisColumn = np.array(embedding_of_thisColumn)
            mean = np.mean(array_of_thisColumn, axis=0)
            
            sum_of_vectors = np.zeros_like(np.outer(mean, mean))
            for v in embedding_of_thisColumn:
                vector = v - mean
                result = np.outer(vector, vector)
                sum_of_vectors += result
            covariance = sum_of_vectors / (length - 1)

            return mean, covariance, length
        else:
            return 0, 0, length

    a_mean, a_covariance, na = find_word2vec_of_one_column(a, model)
    b_mean, b_covariance, nb = find_word2vec_of_one_column(b, model)
    
    if na==0 or nb==0:#如果a和b有一个在fastText里面没找到，那就跳过这个的计算
        return -1
    
    temp = na*nb/(na+nb)
    S = ((na - 1) * a_covariance + (nb - 1) * b_covariance) / (na + nb) / 2
    try:
        inverse_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return -1
    try:
        a_mean_minus_b_mean = a_mean - b_mean
        T_2_ab = temp * np.dot(a_mean_minus_b_mean, np.dot(inverse_S, a_mean_minus_b_mean.T))
        
        #构造F分布， dfn分子自由度， dfd分母自由度，分别是p即词嵌入维度， na+nb-2
        dfn = a_mean.shape[0]
        dfd = na + nb - 2 
        f_dist = stats.f(dfn, dfd)

        #计算F分布在T^2(a,b)处的累积分布函数的值。
        cdf = f_dist.cdf(T_2_ab)
        return cdf
    except AttributeError as e:
        return -1


def u_nl_parrel(a, b, model):

    def find_word2vec_of_one_column(values, model):
        values = [value for value in values if value != " " and value != ""]
        values = [process_string(value) for value in values]
        counter = Counter(values)
        values_set = set(values)

        def process_value(value):
            tokens = value.split()
            embeddings = []
            for token in tokens:
                if token in model.key_to_index:
                    embeddings.append(model[token])
            if len(embeddings) == 0:
                return None

            array = np.array(embeddings)
            average_vector = np.mean(array, axis=0)

            return average_vector, counter[value]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_value, values_set))

        embedding_of_thisColumn = []
        valid_results = [result for result in results if result is not None]
        for result in valid_results:
            average_vector, count = result
            embedding_of_thisColumn.extend([average_vector] * count)

        length = len(embedding_of_thisColumn)
        if length > 1:
            array_of_thisColumn = np.array(embedding_of_thisColumn)
            mean = np.mean(array_of_thisColumn, axis=0)

            sum_of_vectors = np.zeros_like(np.outer(mean, mean))
            for v in embedding_of_thisColumn:
                vector = v - mean
                result = np.outer(vector, vector)
                sum_of_vectors += result
            covariance = sum_of_vectors / (length - 1)

            return mean, covariance, length
        else:
            return 0, 0, length
    
    a_mean, a_covariance, na = find_word2vec_of_one_column(a, model)
    b_mean, b_covariance, nb = find_word2vec_of_one_column(b, model)
    
    if na==0 or nb==0:#如果a和b有一个在fastText里面没找到，那就跳过这个的计算
        return -1
    
    temp = na*nb/(na+nb)
    S = ((na - 1) * a_covariance + (nb - 1) * b_covariance) / (na + nb) / 2
    try:
        inverse_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return -1
    try:
        a_mean_minus_b_mean = a_mean - b_mean
        T_2_ab = temp * np.dot(a_mean_minus_b_mean, np.dot(inverse_S, a_mean_minus_b_mean.T))
        
        #构造F分布， dfn分子自由度， dfd分母自由度，分别是p即词嵌入维度， na+nb-2
        dfn = a_mean.shape[0]
        dfd = na + nb - 2 
        f_dist = stats.f(dfn, dfd)

        #计算F分布在T^2(a,b)处的累积分布函数的值。
        cdf = f_dist.cdf(T_2_ab)
        return cdf
    except AttributeError as e:
        return -1

    
def simhash_lsh(folder_path = 'benchmark', t=2, n = 64):
    hashbits = 64 # simhash签名长度
    num_perm = n  # 哈希函数数量
    lsh = SimhashIndex([], f=num_perm, k=t)#f确定哈希函数数量，k定tolerance，k越大容忍度越高
    idx = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            print(idx)
            idx += 1
            for column in df.columns:
                values = df[column].to_list()
                values = [str(v) for v in values if v != ' ' or v != '']

                texts = " ".join(values)#将每列的每一个值连接成字符串，用空格分开
                
                simhash = Simhash(texts, f=hashbits)
                lsh.add(filename+' '+column, simhash)
    
    return lsh


def u_nl_distribution(lsh, folder_path='benchmark'):
    print("Start calculate Unl distribution.")
    u_sem_all = []
    buckets = lsh.bucket
    num_to_select = 100
    random_keys = random.sample(list(buckets.keys()), num_to_select)
    select_buckets = {key:buckets[key] for key in random_keys}

    embedding_file = 'fastText/wiki-news-300d-1M.vec'
    start_time = time.time()
    model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    np.random.seed(41)
    end_time = time.time()
    print('load fastText time is ' + str((end_time-start_time)/60) +" minutes.")
    idx = 0
    sampling_ratio = 0.05
    for bucket_content in select_buckets.values():
        idx += 1
        print("Start bucket " + str(idx))
        values = list(bucket_content)
        if len(values) < 2:
            continue
        
        values = [v.split(',')[1] for v in values]#取出索引值，即'filename column'的字符串

        combinations = list(itertools.combinations(values, 2))
        length = len(combinations)
        if length<500:
                size = length
        else:
            size = max(1, int(len(combinations)*sampling_ratio))
        size = min(500, length)
        random_numbers = np.random.randint(low=0, high=length, size=size)
        subset = [combinations[i] for i in random_numbers]
        print("bucket"+str(idx)+" size is " + str(len(values)))
        print("bucket"+str(idx)+" combination size is " + str(len(combinations)))
        combineId = 0
        for pair in tqdm(subset):
            file_name1, column_name1 = pair[0].split(maxsplit=1)
            file_name2, column_name2 = pair[1].split(maxsplit=1)
            start_time = time.time()
            column1 = pd.read_csv(folder_path + '/' + file_name1)[column_name1].to_list()
            column2 = pd.read_csv(folder_path + '/' + file_name2)[column_name2].to_list()
            u_nl_score = u_nl(column1, column2, model)
            if u_nl_score != -1:
                u_sem_all.append(u_nl_score)
            combineId += 1
            end_time = time.time()
            run_time = end_time - start_time
            print("process bucket" + str(idx) + " combination"+str(combineId)+' total time is '+str(run_time) +'s.')

        print("End bucket: "+str(idx))
        print()


    # counts = Counter(u_sem_all)

    # count = sum(list(counts.values()))

    # for key in counts:
    #     counts[key] = counts[key]/count    

    return u_sem_all


def u_nl_distribution_parrel(lsh, folder_path='benchmark', n_workers=32, subset_size=1000):
    print("Start calculating Unl distribution.")
    u_sem_all = []
    buckets = lsh.bucket
    embedding_file = 'fastText/wiki-news-300d-1M.vec'
    start_time = time.time()
    model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    np.random.seed(41)
    end_time = time.time()
    print('load fastText time is ' + str((end_time-start_time)/60) + " minutes.")
    
    def process_bucket(bucket_idx, bucket_content):
        values = list(bucket_content)
        if len(values) < 2:
            return
        
        values = [v.split(',')[1] for v in values] # 取出索引值，即 'filename column' 的字符串
        
        combinations = list(itertools.combinations(values, 2))
        length = len(combinations)
        size = min(subset_size, length)
        random_numbers = np.random.randint(low=0, high=length, size=size)
        subset = [combinations[i] for i in random_numbers]
        print("bucket" + str(bucket_idx) + " size is " + str(len(values)))
        print("bucket" + str(bucket_idx) + " combination size is " + str(len(combinations)))
        
        for combineId, pair in enumerate(subset, start=1):
            file_name1, column_name1 = pair[0].split(maxsplit=1)
            file_name2, column_name2 = pair[1].split(maxsplit=1)
            #start_time = time.time()
            column1 = pd.read_csv(folder_path + '/' + file_name1)[column_name1].to_list()
            column2 = pd.read_csv(folder_path + '/' + file_name2)[column_name2].to_list()
            u_nl_score = u_nl(column1, column2, model)
            if u_nl_score != -1:
                u_sem_all.append(u_nl_score)
            # end_time = time.time()
            # run_time = end_time - start_time
            # print("process bucket" + str(bucket_idx) + " combination" + str(combineId) + ' total time is ' + str(run_time) + 's.')
        
        print("End bucket: " + str(bucket_idx))
        print()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_bucket = {executor.submit(process_bucket, bucket_idx, bucket_content): (bucket_idx, bucket_content) for bucket_idx, bucket_content in enumerate(buckets.values(), start=1)}
        for future in concurrent.futures.as_completed(future_to_bucket):
            bucket_idx, _ = future_to_bucket[future]
            try:
                _ = future.result()
            except Exception as exc:
                print('Bucket %d generated an exception: %s' % (bucket_idx, exc))
    
    return u_sem_all


def query_lsh_nl(values, lsh, threshold):
    query = " ".join([str(v) for v in values])
    simhash_query = Simhash(query, f=64)
    results = lsh.get_near_dups(simhash_query)

    return results 


if __name__ == '__main__':
    start_time = time.time()
    # lsh = simhash_lsh()
    # with open("lsh/UnlLSH.pkl", "wb") as f:
    #     pickle.dump(lsh, f)
    with open('lsh/UnlLSH.pkl', 'rb') as f:
        u_sem_lsh = pickle.load(f)
    end_time = time.time()
    run_time = (end_time - start_time) / 60
    print("simHash LSH run time is "+str(run_time) + ' minutes.')
    dic = u_nl_distribution(u_sem_lsh)
    save_dict_to_json(dic, 'distribution/unl_lsh_new.json')
    # print(dic)


    # table_names = os.listdir('benchmark')
    # np.random.seed(41)

    # random_values = np.random.choice(table_names, size = 1, replace= False)

    # query_tables = random_values.tolist()

    # df = pd.read_csv('benchmark/'+query_tables[0])
    # for column in df.columns:
    #     values = df[column].to_list()
    #     query = " ".join(values)
    #     simhash_query = Simhash(query, f=64)
    #     results = simhash_lsh.get_near_dups(simhash_query)
    # for doc_id in results:
    #     print(doc_id)