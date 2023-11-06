import os
import time
from tqdm import tqdm 
import pandas as pd
import pickle

def read_first_column(csv_path):
    # 读取CSV文件，假设第一列是主题列
    df = pd.read_csv(csv_path)
    # 读取第一列数据并存储在一个Series对象中
    first_column_list = df.iloc[:, 0].tolist()
    return first_column_list
def read_attributes(csv_path):
    df = pd.read_csv(csv_path)
    header_list = df.columns.tolist()
    return header_list

def read_sencond_column(csv_path):
    # 读取CSV文件，假设第一列是主题列
    df = pd.read_csv(csv_path)
    # 读取第一列数据并存储在一个Series对象中
    second_column_list = df.iloc[:, 1].tolist()
    return second_column_list
def table_label_value(label_csv_path, n, m):
    df = pd.read_csv(label_csv_path)
    label_value = {}
    count = 0
    for index, row in df.iterrows():
        key = row[1]  # 第二列作为键
        if row[2] == 0 or type(row[2]) == str:
            value = 0
        else:
            value = 1.0 / row[2]  # 第三列作为值
        if key in label_value:
            label_value[key] += value
            count += 1
        else:
            label_value[key] = value
            count += 1
    # print(label_value)
    for key, value in label_value.items():
        label_value[key] = (value ** int(n)) / (count ** int(m))
    return label_value

def can_unique_label(can_label_folder_path, n, m, can_folder_path):
    value_labels = []
    fnames = []
    second_column = []
    all_att = []
    all_can_entity = []
    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(can_label_folder_path)):
        if filename.endswith('.csv'):

            #将filename写入
            fnames.append(filename)

            #获取label文件路径
            file_path = os.path.join(can_label_folder_path, filename)

            #获取候选表的标签和权重
            label_value = table_label_value(file_path, n, m)
            value_labels.append(label_value)

            #读取候选表标签集合，注意这里的路径还是label
            mid = read_sencond_column(file_path)
            column = list(set(mid))
            second_column.append(column)

            #读取候选表的属性表,这里是第一行，这里要用原csv文件
            file_path1 = os.path.join(can_folder_path, filename)
            att = read_attributes(file_path1)
            all_att.append(att)

            #读取候选表的实体列,这里是第一列，这里要用原csv文件
            entity = read_first_column(file_path1)
            all_can_entity.append(list(set(entity)))

    print(len(value_labels))
    print(len(fnames))
    print(len(second_column))
    print(len(all_att))
    print(len(all_can_entity))
    return value_labels, fnames, second_column, all_att, all_can_entity

s = time.time()
file21 = r'/data_ssd/webtable/large/split_1'
file22 = r'/data_ssd/webtable/large/small_query'
file11 = r'../webtable_label/label_folder_split_1'
file12 = r'../webtable_label/label_folder_small_query'
# file1 = r'test_free_result'
# file2 = r'test_free'
can_value_labels1, fnames1, can_second_column1, all_att1, all_can_entity1 = can_unique_label(file11, 2, 2, file21)
can_value_labels2, fnames2, can_second_column2, all_att2, all_can_entity2 = can_unique_label(file12, 2, 2, file22)


can_value_labels1 += can_value_labels2
fnames1 += fnames2
can_second_column1 += can_second_column2
all_att1 += all_att2
all_can_entity1 += all_can_entity2


with open(file='candidate_label_value.pkl',mode='wb') as f:
    pickle.dump(can_value_labels1, f)

with open(file='file_names.pkl',mode='wb') as f:
    pickle.dump(fnames1, f)

with open(file='entity_label.pkl',mode='wb') as f:
    pickle.dump(can_second_column1, f)

# candidate_attributes_list.pkl
with open(file='candidate_attributes_list.pkl', mode='wb') as f:
    pickle.dump(all_att1, f)

with open(file='can_entity_list.pkl', mode='wb') as f:
    pickle.dump(all_can_entity1, f)
e = time.time()
print(e-s)
