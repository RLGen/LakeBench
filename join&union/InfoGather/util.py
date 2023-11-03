import collections
import csv
import re

import numpy as np


def convertArrtibute(name):
    name = name.lower().replace("\n", "")
    # apear \ ()  means multi-attribute name, get the first
    name = re.sub(r'\(.*?\)', "", name)
    name = re.sub(r"\\.*", "", name)
    name = re.sub(r"\s+", "", name)
    return name



## return querytable {colum: {term}}
def getquerydate(file):
    try:
        with open(file, "r", encoding='utf-8') as csvfile:
            reader = csv.read(csvfile)
            headers = next(reader)
            columns = {header: set() for header in headers}
            for row in reader:
                for i, value in enumerate(row):
                    columns[headers[i]].add(value)
        return columns
    except Exception as e:
        print("处理 query 表出现问题%s" % (file))


def random_walk(graph, node, walk_length):
    # 实现随机游走逻辑
    path = [node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(path[-1]))
        if len(neighbors) == 0:
            break
        next_node = np.random.choice(neighbors)
        path.append(next_node)
    return path


def personalized_pagerank(graph, walk_length, num_walks):
    ppr_vectors = collections.defaultdict(dict)
    total = walk_length * num_walks
    for _ in range(num_walks):
        for node in graph.nodes():
            path = random_walk(graph, node, walk_length)
            for visited_node in path:
                if visited_node in ppr_vectors[node].keys():
                    ppr_vectors[node][visited_node] += 1 / total
                else:
                    ppr_vectors[node][visited_node] = 1 / total

    return ppr_vectors


def get_table_value_name(file):
    result_dict = {}

    with open(file, 'r') as file:
        # 使用 csv.reader 读取 CSV 文件
        reader = csv.reader(file)
        # 读取第一行获取列名
        header = next(reader)
        # 初始化字典的值集合
        for col in header:
            result_dict[col] = set()
        # 遍历文件的每一行
        for row in reader:
            # 遍历每一列
            for index, value in enumerate(row):
                col_name = header[index]
                result_dict[col_name].add(value)
    return result_dict


def get_unique_values(csv_file, column_name):
    result_dict = {}
    with open(csv_file, 'r',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            value = row[column_name]
            if column_name in result_dict:
                result_dict[column_name].add(value)
            else:
                result_dict[column_name] = {value}
    return result_dict





def calculate_unionsection(lists):
    # 将第一个列表转为集合，作为初始交集
    intersection_set = lists[0]

    # 遍历剩余列表，计算交集
    for lst in lists[1:]:
        intersection_set = intersection_set.union(lst)

    return intersection_set


def merge_dicts_list(dict_list):
    merged_dict = {}

    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value

    return merged_dict



# process groundTrue.table
def process_csv(file_path):
    """

    Args:
        file_path ():

    Returns:
        是一个字典，key为quary column  value 为 join 的lake table columns list

    """
    result = {}
    with open(file_path, 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key = row[0] + '#' + row[2]
            value = row[1] + '#' + row[3]
            if key in result:
                result[key].add(value)
            else:
                result[key] = {value}
    return result

import os

def scan_directory(path):
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".csv"):
                print("File Name:", entry.name)
                print("File Path:", entry.path)
                print("----------------------")
            elif entry.is_dir():
                scan_directory(entry.path)






