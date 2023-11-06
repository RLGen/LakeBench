import heapq
import pickle
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm
import os
from multiprocessing import Process, Queue
import multiprocessing


#读取表格的第一列，假设第一列是主题列
def read_first_column(df):
    # 读取CSV文件，假设第一列是主题列
    # 读取第一列数据并存储在一个Series对象中
    first_column_list = df.iloc[:, 0].tolist()
    return first_column_list

#读取表格的第二列，假设第一列是主题列
def read_sencond_column(df):
    # 读取CSV文件，假设第一列是主题列
    # 读取第一列数据并存储在一个Series对象中
    first_column_list = df.iloc[:, 1].tolist()
    return first_column_list

#计算一对表格的SECover得分
def one_SECover(list1, list2):  #计算一对表的SECover得分
    # 先将列表的重复值去除，转换为集合
    set1 = set(list1)
    set2 = set(list2)

    #计算set1，set2的元素个数
    len1 = len(set1)

    # 求两个集合的交集，即相同的元素
    intersection_set = set1.intersection(set2)

    # 计算元素相同的个数
    # SECover = len(intersection_set) / len1
    if len1 == 0:
        SECover = 0
    else:
        SECover = len(intersection_set) / len1 
    # print(SECover)
    return SECover

def cal_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2  # 交集运算符
    return intersection

# 计算Jaccard相似度的函数
def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0
    else: 
        return len(intersection) / len(union)

#计算join的值
def find_join(data):
    # 使用max()函数，传入一个自定义的比较函数，基于字典的值来比较
    max_key = max(data, key=data.get)
    f1,f2 = max_key
    return f1,f2


# 计算一个候选表的额外属性
def add_attributes(query_att_list, candidate_att_list):
    # 创建一个无向图
    G = nx.Graph()

    # 将列表1中的元素添加到图的一个集合中
    G.add_nodes_from(query_att_list, bipartite=0)

    # 将列表2中的元素添加到图的另一个集合中
    G.add_nodes_from(candidate_att_list, bipartite=1)

    # 计算并添加边的权重（使用Jaccard相似度）
    for elem1 in query_att_list:
        for elem2 in candidate_att_list:
            weight = jaccard_similarity(elem1, elem2)
            G.add_edge(elem1, elem2, weight=weight)

    # 最大权重匹配
    matching = nx.algorithms.max_weight_matching(G, maxcardinality=True)

    # 输出匹配结果，同时存储相似度值
    matched_pairs = {}
    match_att = []
    for elem1, elem2 in matching:
        similarity = G[elem1][elem2]['weight']
        matched_pairs[(elem1, elem2)] = similarity
        match_att.append(elem2)
    #存储一对一映射的个数，以及list1,list2的元素个数
    # print("映射为",matched_pairs)
    add_attr = [x for x in candidate_att_list if x not in match_att]
    f1, f2 = find_join(matched_pairs)
    return add_attr, f1, f2



#读取表格第一行,即读取表头
def read_att(mytable_path):
    # 读取CSV文件，假设第一行是表头
    df = pd.read_csv(mytable_path,low_memory=False, lineterminator="\n")

    # 获取表头，并存放在一个列表中
    headers_list = df.columns.tolist()
    return headers_list


def one_SSB(query_att, one_table_add_att, one_att_freq, two_att_freq):
    end_dict = {}
    length = len(query_att)
    for add in one_table_add_att:
        count = 0.0
        for query in query_att:
            mid = []
            mid.append(add)
            mid.append(query)
            t1 = tuple(mid)
            if t1 in two_att_freq:
                mid1 = two_att_freq[t1]
                # print(mid1)
            else:
                mid1 = 0

            if query in one_att_freq:
                # print(query)
                mid2 = one_att_freq[query]

            else:
                mid2 = 0
            # print(mid1, mid2)
            if mid2 == 0:
                mid_result = 0
            else:
                mid_result = mid1 / mid2
            count += mid_result
        if length == 0:
            end_dict[add] = 0
        else:
            end_dict[add] = count / length

    if end_dict:
        max_item = max(end_dict.items(), key=lambda item: item[1])
        result = max_item[1]
    else:
        result = 0
    return result

def find_largest_five(my_dict, top_k):
    files_result = []
    largest_items = heapq.nlargest(top_k, my_dict.items(), key=lambda item: item[1])
    for key, value in largest_items:
        files_result.append(key)
    # print(files_result)
    return files_result

def all_schema(fnames, can_label_second_column, query_label_second_column, can_entity_list, query_entity,
               query_attribute, can_all_att, one_att_freq, two_att_freq, top_k):
    count = 0
    result = {}
    join = {}
    for file_name in fnames:
        second = can_label_second_column[count]
        inter = cal_intersection(second, query_label_second_column)

        if inter == 0:
            result[file_name] = 0.0
            count += 1
        else:
            SEcover = one_SECover(query_entity, can_entity_list[count])
            add, f1, f2= add_attributes(query_attribute, can_all_att[count])

            mid_f = []
            mid_f.append(f1)
            mid_f.append(f2)

            join[file_name] = mid_f

            SSB = one_SSB(query_attribute, add, one_att_freq, two_att_freq)
            result[file_name] = SEcover * SSB
            count += 1

    end_result = find_largest_five(result, top_k)
    join_dict = []

    for item in end_result:
        join_dict.append(join[item])

    return end_result, join_dict




def query_online(query_folder_path, query_label_folder_path, fnames, can_label_second_column, can_entity_list,
can_all_att, one_att_freq, two_att_freq, top_k):
    all_query_result = []
    query_csv_files = os.listdir(query_folder_path)
    join = []
    for filename in tqdm(query_csv_files):
        query_label_path = os.path.join(query_label_folder_path, filename)

        query_label_df = pd.read_csv(query_label_path, low_memory=False, lineterminator='\n')

        query_path =  os.path.join(query_folder_path, filename)

        query_df = pd.read_csv(query_path, low_memory=False, lineterminator='\n')

        query_attribute = read_att(query_path)


        query_label_second_column = read_sencond_column(query_label_df)

        query_entity = read_first_column(query_label_df)


        result, join_dict = all_schema(fnames, can_label_second_column, query_label_second_column, can_entity_list, query_entity,
               query_attribute, can_all_att, one_att_freq, two_att_freq, top_k)

        count = 0
        for item in result:
            mid = []
            mid_list = join_dict[count]
            count += 1
            mid.append(filename)
            mid.append(item)
            mid.append(mid_list[0])
            mid.append(mid_list[1])
            join.append(mid)
        # join.append(join_dict)
        # print(join_dict)

    return join




        # for value in result.values():
        #     join_list1 = []
        #     join_list1.append(filename)
        #     join_list1.append(value)

        #     # join.append(filename)
        #     index = -1
        #     if value in my_list:
        #         index = my_list.index(fnames)
        #     can_att = can_all_att[index]
        #     max_att1, max_att2 = can_pairs(query_attribute, can_att)
        #     join_list1.append(max_att1)
        #     join_list1.append(max_att2)
        #     join.append(join_list1)
        
    # return join 



    
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



# def find_join(can_file_path, query_file_path):
#     can_att = read_att(can_file_path)
#     query_att = read_att(query_file_path)
#     max_att = can_pairs(query_att, can_att)
#     return max_att
    
# def csv_make(save_csv_path, all_join_result):

    # with open(save_csv_path, mode='w', newline='') as csv_file:
    # # 创建CSV写入对象
    # csv_writer = csv.writer(csv_file)
    
    # # 遍历列表l中的每个子列表，并将它们写入不同的行
    # for join_result in all_join_result:
    #     csv_writer.writerow(join_result)
    # print(f"已将列表写入到 {csv_file_path} 文件中。")
    





#获取文件名
fnames = []
files_names = r'../offline_processing_webtable/file_names.pkl'
with open(files_names, 'rb') as f:
    fnames = pickle.load(f)


#获取候选表的实体标签
#with open(file='entity_label.pkl',mode='wb') as f:
    # pickle.dump(second_column, f)
can_label_second_column = []
can_entity_label = r'../offline_processing_webtable/entity_label.pkl'
with open(can_entity_label, 'rb') as f:
    can_label_second_column = pickle.load(f)




#获取候选表所具有的实体
can_entity = r'../offline_processing_webtable/can_entity_list.pkl'
can_entity_list = []
with open(can_entity, 'rb') as f:
    can_entity_list = pickle.load(f)


#获取候选表的属性总列表
can_all_att = []
with open('../offline_processing_webtable/candidate_attributes_list.pkl', 'rb') as f:
    can_all_att = pickle.load(f)

#获取属性出现的频数
one_att_freq = []
with open('../offline_processing_webtable/one_att_freq.pkl', 'rb') as f:
    one_att_freq = pickle.load(f)

two_att_freq = []
with open('../offline_processing_webtable/two_att_freq.pkl', 'rb') as f:
    two_att_freq = pickle.load(f)


# query_folder_path = r'/data_ssd/opendata/small/query/'
# query_label_folder_path = r'../offline_opendata/query'
query_folder_path = r'/data_ssd/opendata/small/query/'
query_label_path = r'../label_webtable/query'

# file_list = os.listdir(query_folder_path)
top_k = 25


# query_online(query_folder_path, query_label_folder_path, fnames, can_label_second_column,
# can_entity_list, can_all_att, one_att_freq, two_att_freq, top_k)

all_query_result = []
query_csv_files = os.listdir(query_folder_path)

s = time.time()

def multi(file_ls,queue,query_result,query_folder_path, query_label_folder_path, fnames, can_label_second_column,
 can_entity_list, can_all_att, one_att_freq, two_att_freq,top_k):
    join = []
    for filename in file_ls:
        query_label_path = os.path.join(query_label_folder_path, filename)

        query_label_df = pd.read_csv(query_label_path, low_memory=False, lineterminator='\n')

        query_path =  os.path.join(query_folder_path, filename)

        query_df = pd.read_csv(query_path, low_memory=False, lineterminator='\n')

        query_attribute = read_att(query_path)


        query_label_second_column = read_sencond_column(query_label_df)

        query_entity = read_first_column(query_label_df)


        result, join_dict = all_schema(fnames, can_label_second_column, query_label_second_column, can_entity_list, query_entity,
                query_attribute, can_all_att, one_att_freq, two_att_freq, top_k)

        count = 0
        for item in result:
            mid = []
            mid_list = join_dict[count]
            count += 1
            mid.append(filename)
            mid.append(item)
            mid.append(mid_list[0])
            mid.append(mid_list[1])
            join.append(mid)

        queue.put(1)
    query_result.put(join)
    queue.put((-1, "test-pid"))


split_num = 30
query_result = multiprocessing.Manager().Queue()


sub_file_ls = split_list(query_csv_files, split_num)

process_list = []

#####
# 为每个进程创建一个队列
queues = [multiprocessing.Manager().Queue() for i in range(split_num)]
# queue = Queue()
# 一个用于标识所有进程已结束的数组
finished = [False for i in range(split_num)]

# 为每个进程创建一个进度条
bars = [tqdm(total=len(sub_file_ls[i]), desc=f"bar-{i}", position=i) for i in range(split_num)]
# bar = tqdm(total=len(file_ls[0]), desc=f"process-{i}")
# 用于保存每个进程的返回结果
results = [None for i in range(split_num)]

for i in range(split_num):
    process = Process(target=multi, args=(sub_file_ls[i], queues[i], query_result, query_folder_path, query_label_folder_path, fnames, can_label_second_column,
 can_entity_list, can_all_att, one_att_freq, two_att_freq,top_k))
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

final_result = []
while not query_result.empty():
    try:
        element = query_result.get_nowait()
        final_result.extend(element)
    except Exception as e:
        continue


# 指定CSV文件路径
csv_file_path = './new_webtable' + str(top_k) + '.csv'

# 将二维列表写入CSV文件
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # 遍历二维列表的每一行，写入CSV文件
    for row in final_result:
        csv_writer.writerow(row)

e = time.time()
print(e-s)