import numpy as np
import pandas as pd

def calculate_metrics(query_table, df_result, df_gt):
    #取出表名=query_table的那些行
    df_result_of_q = df_result[df_result['query_table'] == query_table]
    df_gt_of_q = df_gt[df_gt['query_table'] == query_table]

    #求结果和ground_truth的交集
    set_result = set(df_result_of_q.itertuples(index = False, name=None))
    set_gt = set(df_gt_of_q.itertuples(index = False, name=None))

    intersection = set_result.intersection(set_gt)
    
    #交集转回DataFrame
    #intersection_df = pd.DataFrame(list(intersection), columns=df_result.columns)

    #计算精度召回
    len_intersection = len(intersection)
    len_result = df_result_of_q.shape[0]
    len_gt = df_gt_of_q.shape[0]
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


if __name__ == '__main__':
    groundtruth_path = 'groundtruth/alignment_groundtruth.csv'
    result_path = 'tusResult/alignment_result.csv'
    df_gt = pd.read_csv(groundtruth_path)
    df_result = pd.read_csv(groundtruth_path)
    column_data = df_result["query_table"]
    unique_values = column_data.unique()
    #对结果里的每个表计算精度，召回率，以及map
    precisions = []
    recalls = []
    for table in unique_values:
        precision, recall = calculate_metrics(table, df_gt, df_result)#计算这一个表的精度召回
        precisions.append(precision)
        recalls.append(recall)
    
    precision = np.mean(precisions)
    recall = np.mean(recalls)

    print("Precision:", precision)
    print("Recall:", recall)
