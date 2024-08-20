#根据fastText的结果，计算出每一列对应的向量表示，也是存到另一个文件夹，'benchmarkNL'
import os
import numpy as np
import pandas as pd

from collections import Counter
from gensim.models import KeyedVectors
from yago import process_string
from Usem import fill_zero


def find_word2vec_of_column():
    #加载fasttext模型
    embedding_file = 'fastText/wiki-news-300d-1M.vec'
    model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    
    #加载所有的表
    folder_path = 'benchmark'
    save_folder_path = 'benchmarkNl'
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            saved_data = {}

            for column in df.columns:
                values = df[column].to_list()
                values = [value for value in values if value!=" " and value!=""]
                values = [process_string(value) for value in values]
                counter = Counter(values)

                #统计值出现的次数，形成一个字典
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
                    average_vector = np.mean(array, axis = 0)
                    average_list = average_vector.tolist()
                    string_list = ' '.join(str(s) for s in average_list)
                    
                    #添加对应个数的值
                    for i in range(counter[value]):
                        embedding_of_thisColumn.append(string_list)


                saved_data[column] = embedding_of_thisColumn
            
            saved_data = fill_zero(saved_data)
            saved_df = pd.DataFrame(saved_data)
            saved_file_path = os.path.join(save_folder_path, file_name)
            saved_df.to_csv(saved_file_path, index=False)
            print('sucess saved on ' + saved_file_path)


if __name__ == '__main__':
    find_word2vec_of_column()