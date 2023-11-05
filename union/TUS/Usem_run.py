#运行生成表的临时文件
import os
import sqlite3
import pandas as pd
import numpy as np

from collections import Counter
from yago import get_yagoTypes_from_tsv, process_string


#确保DataFrame中所有列的长度相同，短的列填0
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


def match_entity_from_yago():
    
    #创建虚拟表，用以执行全文搜索
    conn, cursor = get_yagoTypes_from_tsv()

    #先一把所有表的所有行都映射到yago中的类，使用match，全文搜索
    floder_path = 'benchmark'
    save_floder_path = 'benchmarkSem'
    for file_name in os.listdir(floder_path):
        if file_name not in os.listdir('benchmarkSem'):
            file_path = os.path.join(floder_path, file_name)
            df = pd.read_csv(file_path)
            saved_data = {}
            #取出每一列，将列的每个值都先映射到实体，选出最多三个实体，把这个实体映射到类中
            for column in df.columns:
                values = df[column].to_list()
                #把所有的值都处理成小写，并用空格来代替标点符号
                for i in range(len(values)):
                    values[i] = process_string(values[i])
                values = [s for s in values if s!=" " and s!=""]
                counter = Counter(values)
                #统计值出现的次数，形成一个字典
                values_set= set(values)
                classes_of_thisColumn = []
                for value in values_set:
                    search_keywords = value
                    query = f'''select entity
                    from yagoEntitys 
                    where entity match '{search_keywords}' 
                    order by bm25(yagoEntitys) desc 
                    limit 3;'''
                    try:
                        cursor.execute(query)
                        results = cursor.fetchall()
                    #找到了这个value匹配的实体名称，还要依次把相应的taxonomy_class类找出来
                        for result in results:
                            query = f'''SELECT distinct taxonomy_class
                            FROM yagoTypejoinclasses
                            WHERE entity = '{result[0]}';
                            '''
                            cursor.execute(query)
                            temp = cursor.fetchall()
                            for i in range(len(temp)):
                                temp[i] = temp[i][0]
                            
                            #根据列中value出现的次数，添加相应个数的class值
                            classes_of_thisColumn += temp * counter[value]#列表*一个数，是把列表扩大几倍
                            #查询返回的是一个元组，需要把元组中的[0]拿出来，然后放进classes_of_thisColumn, 并且需要distinct
                    except sqlite3.Error:
                        continue
                
                #把这个列的属于的所有类都存到字典中
                saved_data[column] = classes_of_thisColumn
            
            #创建DataFrame对象，把saved_data字典加载到这里面，最后使用to_csv将这个表对应的sem表保存
            saved_data = fill_zero(saved_data)
            saved_df = pd.DataFrame(saved_data)
            saved_file_path = os.path.join(save_floder_path, file_name)
            saved_df.to_csv(saved_file_path, index=False)
            print("sucess saved on " + saved_file_path)

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    match_entity_from_yago()


