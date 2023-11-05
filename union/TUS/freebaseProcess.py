import os
import json

import pickle
import pandas as pd
import numpy as np
from collections import Counter
from Usem import query_lsh 
from yago import get_yagoTypes_from_tsv, process_string
from datasketch import MinHash, MinHashLSH

#把yago里的实体插入lsh，并且把表中的unique的cell values插入到lsh
# yago里的实体的标签就是字符串‘实体， 所属的类， 类包含多少实体’
#cell values的标签就是cell values的值
#通过这个lsh索引分桶里面选一个实体作为这个桶里所有值匹配的实体，阈值设为0.9，然后建个字典，存起来，处理表的时候直接查

def entity_cellvalues_lsh(folder_path = 'benchmark', yago_path = 'yago', file_name='yagoTypes.tsv', threshold=0.95):
    #file_path = os.path.join(yago_path, file_name)
    lsh = MinHashLSH(threshold=0.95, num_perm=128)
    entity_set = set()
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         _, entity, relation, t = line.strip().split('t')
    #         entity = process_string(entity[1:-1])
    #         entity_set.add(entity)
    conn, cursor = get_yagoTypes_from_tsv()
    query = "SELECT distinct entity, taxonomy_class, entity_count FROM yagoTypejoinclasses"
    cursor.execute(query)

    # 获取查询结果
    results = cursor.fetchall()
    for row in results:
        minhash = MinHash(num_perm=128)
        minhash.update(row[0].encode('utf-8'))
        index = "+-*/".join(row)
        lsh.insert(index, minhash)


def match_yago_entity_from_yago_lsh(lsh, file_subset):
    floder_path = 'benchmark'
    save_floder_path = 'freebaseResult'
    conn, cursor = get_yagoTypes_from_tsv()
    exist_list = os.listdir(save_floder_path)
    cal_list = [x for x in file_subset if x not in exist_list]
    for file_name in cal_list:
        if file_name in cal_list:
            if file_name.endswith('.csv'):
                file_path = os.path.join(floder_path, file_name)
                df = pd.read_csv(file_path)
                saved_data = {}

                column = df.columns[0]
                values = df[column].to_list()
                values = [process_string(s) for s in values if s!='' and s!=' ']
                counter = Counter(values)
                values_set = set(values)

                entity_of_thisColumn = []
                classes_of_thisColumn = []
                domin_of_thisColumn = []
                for value in values_set:
                    search_keywords = value
                    result = query_lsh(lsh, search_keywords, 1)

                    if len(result) > 0:
                        query = f'''SELECT entity, taxonomy_class, entity_count
                        FROM yagoTypejoinclasses
                        where entity = '{result[0]}'
                        '''

                        cursor.execute(query)
                        temp = cursor.fetchall()
                        for t in temp:
                            for i in range(counter[value]):
                                entity_of_thisColumn.append(t[0])
                                classes_of_thisColumn.append(t[1])
                                domin_of_thisColumn.append(t[2])

                saved_data['entity'] = entity_of_thisColumn
                saved_data['classes'] = classes_of_thisColumn
                saved_data['domin'] = domin_of_thisColumn
                saved_df = pd.DataFrame(saved_data)
                save_file_path = os.path.join(save_floder_path, file_name)
                saved_df.to_csv(save_file_path, index=False)
                print('sucess saved on ' + save_file_path)
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    with open("lsh/yagoLSH.pkl", "rb") as f:
        lsh = pickle.load(f)
    match_yago_entity_from_yago_lsh(lsh, os.listdir('benchmark'))

