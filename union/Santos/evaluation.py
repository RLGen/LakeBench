import pickle
import json
import numpy as np
import pandas as pd

def calculate_metrics(query_table, df_result, df_gt):
    #取出表名=query_table的那些行
    df_result_of_q = df_result[df_result['query_table'] == query_table]
    df_gt_of_q = df_gt[df_gt['query_table'] == query_table]

    #求结果和ground_truth的交集
    set_result = set(df_result_of_q.itertuples(index = False, name=None))
    selected_columns = df_gt_of_q.iloc[:, :2]
    set_gt = set(map(tuple, selected_columns.values))

    #set_gt = set(df_gt_of_q.itertuples(index = False, name=None))

    intersection = set_result.intersection(set_gt)
    
    #交集转回DataFrame
    #intersection_df = pd.DataFrame(list(intersection), columns=df_result.columns)

    #计算精度召回
    len_intersection = len(intersection)
    len_result = len(set_result)
    len_gt = len(set_gt)
   
    precision = len_intersection/len_result
    recall = len_intersection/len_gt

    return precision, recall, len_intersection, len_result, len_gt


def calculate_metrics_dict(query_table, result_list, gt_list):
    #求结果和ground_truth的交集
    set_result = set(result_list)
    set_gt = set(gt_list)

    intersection = set_result.intersection(set_gt)
    
    #交集转回DataFrame
    #intersection_df = pd.DataFrame(list(intersection), columns=df_result.columns)

    #计算精度召回
    len_intersection = len(intersection)
    len_result = len(result_list)
    len_gt = len(gt_list)
    precision = len_intersection/len_result
    recall = len_intersection/len_gt

    return precision, recall


def calculate_average_precision(actual, predicted):
    # 计算平均精度（AP）的辅助函数
    num_hits = 0
    sum_precision = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            sum_precision += precision_at_i

    if not actual:
        return 0.0

    average_precision = sum_precision / len(actual)
    return average_precision


def calculate_mean_average_precision(actual_list, predicted_list):
    # 计算平均精度MAP
    average_precisions = []
    for actual, predicted in zip(actual_list, predicted_list):
        average_precision = calculate_average_precision(actual, predicted)
        average_precisions.append(average_precision)

    mean_average_precision = sum(average_precisions) / len(average_precisions)
    return mean_average_precision


def save_dict_to_json(data, path):
    for key, value in data.items():
        if isinstance(value, set):
            data[key] = list(value) 
    with open(path, 'w') as json_file:
        json.dump(data, json_file)


def evalute_csv(which_benchmark, which_method, topk):
    if which_benchmark == 0:
        current_benchmark = 'webtable'
        groundtruth_path = '/data_ssd/webtable/small_query/ground_truth.csv'
    elif which_benchmark == 1:
        current_benchmark = 'opendata'
        groundtruth_path = '/data_ssd/opendata/small_query/ground_truth.csv'
    elif which_benchmark == 2:
        current_benchmark = 'webtable_large'
        groundtruth_path = '/data_ssd/webtable/large_query/ground_truth.csv'
    elif which_benchmark == 3:
        current_benchmark = 'opendata_large'
        groundtruth_path = '/data/opendata/large_query/ground_truth.csv'
    
    if which_method == 0:
        current_method = 'santos'
    else:
        current_method = 'tus'
    
    #result_path = 'result/webtable_benchmark_true_result_by_santos_full_20.csv'
    result_path = 'result1/' + current_benchmark + '_' + current_method + '_top' + str(topk) + '.csv'
    df_gt = pd.read_csv(groundtruth_path)
    df_result = pd.read_csv(result_path)
    column_data = df_result["query_table"]
    unique_values = column_data.unique()
    #对结果里的每个表计算精度，召回率，以及map
    precisions = []
    recalls = []
    save_dict = {}
    #保存一个字典，key是table name，value是list精度召回，存成json
    for table in unique_values:
        precision, recall, len_intersection, len_result, len_gt = calculate_metrics(table, df_result, df_gt)#计算这一个表的精度召回
        precisions.append(precision)
        recalls.append(recall)
        save_dict[table] = (precision ,recall, len_intersection ,len_result, len_gt)
    
    
    precision = np.mean(precisions)
    recall = np.mean(recalls)

    save_dict_to_json(save_dict, 'result/webtable_evaluation_5.json')
    print("Precision:", precision)
    print("Recall:", recall)


def evalute_dict():
    groundtruth_path = '../groundtruth/santosUnionBenchmark.pickle'
    result_path = '../stats/santos_benchmark_result_by_santos_full.pickle'
    with open(result_path, 'rb') as f:
        result_dict = pickle.load(f)

    with open(groundtruth_path, 'rb') as f:
        gt_dict = pickle.load(f)
    
    precisions = []
    recalls = []
    for table in result_dict.keys():
        precision, recall = calculate_metrics_dict(table, result_dict[table], gt_dict[table])#计算这一个表的精度召回
        precisions.append(precision)
        recalls.append(recall)
    
    precision = np.mean(precisions)
    recall = np.mean(recalls)

    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == '__main__':
    which_benchmark = 1
    which_method = 1
    topk = 60
    evalute_csv(which_benchmark, which_method, topk)
