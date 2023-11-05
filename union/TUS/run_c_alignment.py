import os
import pdb
import json
import time
import pickle
import logging
import itertools
import networkx as nx
import pandas as pd
import numpy as np

from collections import Counter
from gensim.models import KeyedVectors
from Uset import u_set_or_sem, read_distribution, query_lsh, cal_precise_unionablity, minhash_Lsh
from Unl import u_nl, query_lsh_nl, simhash_lsh, cumulative_frequency, save_dict_to_json
from concurrent.futures import ProcessPoolExecutor
from datasketch import MinHash
from operator import itemgetter

def get_candidate(table_name, lsh, folder_path, topk, Aunion, model, type='Uset', threshold=0.7):#改调用query_lsh的方式
    file_path = os.path.join(folder_path, table_name)
    df = pd.read_csv(file_path)
    for column in df.columns:            
        values = df[column].to_list()
        if type == 'Uset' or type == 'Usem':
            results = query_lsh(values, lsh, type, n=128, folder_path=folder_path)
        else:
            results = query_lsh_nl(values, lsh, threshold=threshold)
        #对所有的LSH候选结果计算精确的结果
        set_col_dict = cal_precise_unionablity(type, values, results, folder_path, model)

        #把col_dict按照values的大小排序，倒序，把前几个放到集合A_union中
        sorted_col_dict = dict(sorted(set_col_dict.items(), key=itemgetter(1), reverse=True))

        #这里要不要筛选一遍？还是把所有的候选表都放进去，算表的可并性的时候再筛选
        if len(sorted_col_dict)>=topk:
            Aunion.update(list(sorted_col_dict.keys())[:topk])#这里是选top5个
        else:
            Aunion.update(list(sorted_col_dict.keys()))
#Usem的minhashLSH索引得改，改成最后一行插入的

def get_column_goodness(score, type = 'uset'):
    file_path = os.path.join('distribution', type + '_lsh_730.json')
    freq = cumulative_frequency(file_path, score)
    return freq


def compute_ensemble_score(query_column, candiate_column, query_column_sem, candiate_column_sem, model):
    #计算三个的精确结果，查分布，得到得分
    domin = {'uset', 'usem', 'unl'}
    u_nl_score = u_nl(query_column, candiate_column, model)
    u_set_score = u_set_or_sem('Uset', query_column, candiate_column)
    u_sem_score = u_set_or_sem('Usem', query_column_sem.to_list(), candiate_column_sem.to_list())
    scores = [u_nl_score, u_set_score, u_sem_score]
    
    u_ensemble = 0
    for type,score in zip(domin, scores):
        u_ensemble = max(u_ensemble, get_column_goodness(score, type=type))
    
    return u_ensemble


def get_c_goodness(c, max_c_score):
    file_path = os.path.join('result', str(c)+'_align_distribution.json')
    with open('result/uset_lsh_all.json', 'r') as file:
        dic = json.load(file)
    goodness = 0
    for key, values in dic.items():
        if float(key) <= max_c_score:
            goodness += values

    return goodness


def alignment(query, candiate, type = 'cal'):
    #query和candiate分别是两个表，然后需要提出他们的column当作节点
    df_query = pd.read_csv('benchmark/' + query)
    df_cand = pd.read_csv('benchmark/' + candiate)
    df_query_sem = pd.read_csv('UsemLshTest/' + query)
    df_cand_sem = pd.read_csv('UsemLshTest/'+ candiate)
    G = nx.Graph()
    query_columns = list(df_query.columns)
    query_columns = [query+' '+l for l in query_columns]
    cand_columns = list(df_cand.columns)
    cand_columns = [candiate+' '+l for l in cand_columns]
    G.add_nodes_from(query_columns, bipartite=0)
    G.add_nodes_from(cand_columns, bipartite=1)
    max_c = min(len(query_columns), len(cand_columns))#最大的c
    for attr_s in query_columns:
        for attr_t in cand_columns:
            s_name = attr_s.split(maxsplit=1)[1]
            t_name = attr_t.split(maxsplit=1)[1]
            score = compute_ensemble_score(df_query[s_name], df_cand[t_name], df_query_sem[s_name], df_cand_sem[t_name])
            G.add_edge(attr_s, attr_t, weight=score)

    matching = {}
    edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    max_c_scores = {}
    c = 1
    mul = 1
    for u, v, d in edges:
        if u in matching or v in matching.values():
            continue
        matching[u] = [v, d['weight']]
        mul = mul * d['weight']
        max_c_scores[c] = mul
        c += 1
        if len(matching) == max_c:
            break
    if type == 'distribution':
        return max_c_scores
    
    goodness_score = 0
    best_c = 1
    for c, max_c_score in max_c_scores.items():
        c_goodness = get_c_goodness(c, max_c_score)
        if  c_goodness >= goodness_score:
            goodness_score = c_goodness
            best_c = c
    sorted_matching = sorted(matching.items(), key = lambda item : item[1][1], reverse=True)
    
    return sorted_matching[:best_c], goodness_score


def alignment_process(query, candiate, model, type = 'cal'):
    #query和candiate分别是两个表，然后需要提出他们的column当作节点
    def compute_score(args):
        s_name, t_name, s, t, s_sem, t_sem, m = args
        logging.info(f"Processing: {s_name, t_name}")
        score = compute_ensemble_score(s, t, s_sem, t_sem, m)
        return s_name, t_name, score

    df_query = pd.read_csv('benchmark/' + query)
    df_cand = pd.read_csv('benchmark/' + candiate)
    df_query_sem = pd.read_csv('UsemLshTest/' + query)
    df_cand_sem = pd.read_csv('UsemLshTest/'+ candiate)
    G = nx.Graph()
    query_columns = list(df_query.columns)
    query_columns = [query+' '+l for l in query_columns]
    cand_columns = list(df_cand.columns)
    cand_columns = [candiate+' '+l for l in cand_columns]
    G.add_nodes_from(query_columns, bipartite=0)
    G.add_nodes_from(cand_columns, bipartite=1)
    max_c = min(len(query_columns), len(cand_columns))#最大的c

    task_list = []
    for attr_s in query_columns:
        s_name = attr_s.split(maxsplit=1)[1]
        s = df_query[s_name]
        s_sem = df_query_sem[s_name]
        for attr_t in cand_columns:
            t_name = attr_t.split(maxsplit=1)[1]
            t = df_cand[t_name]
            t_sem = df_cand_sem[t_name]

            task_list.append((attr_s, attr_t, s, t, s_sem, t_sem, model))
    
    with ProcessPoolExecutor(max_workers = 4) as executor:
        results = executor.map(compute_score, task_list)
    
    for s_name, t_name, score in results:
        G.add_edge(s_name, t_name, weight=score)

    matching = {}
    edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    max_c_scores = {}
    c = 1
    mul = 1
    for u, v, d in edges:
        if u in matching or v in matching.values():
            continue
        matching[u] = [v, d['weight']]
        mul = mul * d['weight']
        max_c_scores[c] = mul
        c += 1
        if len(matching) == max_c:
            break
    if type == 'distribution':
        return max_c_scores
    
    goodness_score = 0
    best_c = 1
    for c, max_c_score in max_c_scores.items():
        c_goodness = get_c_goodness(c, max_c_score)
        if  c_goodness >= goodness_score:
            goodness_score = c_goodness
            best_c = c
    sorted_matching = sorted(matching.items(), key = lambda item : item[1][1], reverse=True)
    
    return sorted_matching[:best_c], goodness_score


def cal_max_c_alignment(model, folder_path = 'benchmark'):
    file_names = os.listdir(folder_path)
    #找出表中最大的列数
    max_c = 0
    for file_name in file_names:
        df = pd.read_csv(folder_path+'/'+file_name)
        max_c = max(max_c, len(df.columns))
    #使用字典存每个max_c的分布，key是c，value是列表
    max_c_score_dict = {}
    for i in range(max_c):
        max_c_score_dict[i+1] = []

    np.random.seed(41)
    combinations = list(itertools.combinations(file_names, 2))
    random_numbers = np.random.randint(low=0, high=len(combinations), size=800)#11696
    subset = [combinations[i] for i in random_numbers]
    idx = 0
    for pair in subset:
        d = alignment_process(pair[0], pair[1], model, 'distribution')
        for c, score in d.items():
            max_c_score_dict[c].append(score)
        idx += 1
        print(idx)

    for c, l in max_c_score_dict.items():
        file_path = 'alignDistribution/'+str(c)+'_align_distribution.json'
      
        save_dict_to_json(l, file_path)

#首先读取所有的表，循环处理，每次处理一个表
def main():
    start_time = time.time()
    with open("lsh/UsetLSH.pkl", "rb") as f:
        u_set_lsh = pickle.load(f)
    with open("lsh/UsemLSH.pkl", "rb") as f:
        u_sem_lsh = pickle.load(f)
    with open("lsh/UnlLSH.pkl", "rb") as f:
        u_nl_lsh = pickle.load(f)
    
    df = pd.read_csv("groundtruth/att_groundtruth.csv")
    column_data = df["query_table"]
    unique_values = column_data.unique()
    np.random.seed(41)
    random_values = np.random.choice(unique_values, size=100, replace=False)

    query_tables = random_values.tolist()
    end_time = time.time()
    att_columns = ['query_table', 'candidate_table', 'query_col_name', 'candidate_col_name']
    alignment_columns = ['query_table', 'candidate_table', 'c']
    df_att = pd.DataFrame(columns=att_columns)
    df_align = pd.DataFrame(columns=alignment_columns)
    df_att.to_csv('tusResult/att_result.csv') 
    df_align.to_csv('tusResult/alignment_result.csv')
    print('pre computed time: ' + str(end_time - start_time)+'s.')
    #对每个query table求候选表
    for table in query_tables:
        #分别加载
        #对每一列求候选的表格，所有候选表去重
        start_time = time.time()
        Aunion = set()
        #把这一段写成函数的形式
        get_candidate(table, u_set_lsh, 'benchmark', 5, Aunion, model, type='Uset', threshold=threshold)
        time1 = time.time()
        print('get uset candiate time: '+str(time1 - start_time)+'s.')

        get_candidate(table, u_sem_lsh, 'UsemLshTest', 5, Aunion, model, type='Usem', threshold=threshold)
        time2 = time.time()
        print('get usem candiate time: '+str(time2 - time1)+'s.')

        get_candidate(table, u_nl_lsh, 'benchmark', 5, Aunion, model, type = 'Unl', threshold=threshold)
        time3 = time.time()
        print('get unl candiate time: '+str(time3 - time2)+'s.')
        

        #求出了Aunion,之后根据Aunion计算准确地Uensem，然后根据这个淘汰一部分，最后计算topk个表
        candiate_table_set = []
        for candiate_table in Aunion:
            matching, score = alignment(table, candiate_table, 'cal')
            candiate_table_set.append((matching, score))
        
        end_time = time.time()
        total_align_time = end_time-time3
        sorted_candiate_table_set = sorted(candiate_table_set, key = lambda x:x[1], reversed=True)

        print(table+' use time '+str(total_align_time)+' s.'+' average process a candiate time is '+str(total_align_time/len(Aunion)))
        att_save = {}
        for column in att_columns:
            att_save[column] = []

        align_save = {}
        for column in alignment_columns:
             align_save[column] = []

        for i in range(topk):
            temp = sorted_candiate_table_set[i][0]
            #保存att
            candiate_table_now = temp[0][1][0].split(maxsplit=1)[0]
            for m in temp:
                att_save['query_table'].append(table)
                att_save['candidate_table'].append(candiate_table_now)
                att_save['query_col_name'].append(m[0].split(maxsplit=1)[1])
                att_save['candidate_col_name'].append(m[1][0].split(maxsplit=1)[1])
            #保存align
            align_save['query_table'].append(table)
            align_save['candiate_table'].append(candiate_table_now)
            align_save['c'].append(str(len(temp)))

        #通过pandas保存 
        df_att = pd.DataFrame(att_save)
        df_att.to_csv('tusResult/att_resut.csv', mode='a', header=False, index=False)
        
        df_align = pd.DataFrame(alignment)
        df_align.to_csv('tusResult/alignment_resut.csv', mode='a', header=False, index=False)

#改成多线程        


#统计一下每个模块的时间，是哪里出了问题，输出候选表的个数，alignment算一个的时间


if __name__ == "__main__":
    threshold = 0.7
    embedding_file = 'fastText/wiki-news-300d-1M.vec'
    model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    topk = 30
    # main()
    cal_max_c_alignment(model)
    #完善main函数，可以写输入输出了