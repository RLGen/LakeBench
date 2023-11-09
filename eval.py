import pandas as pd
import numpy as np
import csv

k = 20

gtPath = './gt_.csv'

df = pd.read_csv("/starmie/webtable/result/small/top_20.csv", header=None)
# df = pd.read_csv("/home/dengyuhao/gt_webtable_.csv")
df.drop(df.columns[4], axis=1, inplace=True)
df.drop(df.columns[3], axis=1, inplace=True)
df.drop(df.columns[2], axis=1, inplace=True)
df.columns = ['query_table', 'candidate_table']

grouped = df.groupby(df.columns[0])
gt_df = pd.read_csv(gtPath)

# print(df.head(5))
# print(gt_df.head(5))

# print(gt_df)
# gt_df.drop(gt_df.columns[3], axis=1, inplace=True)
# gt_df.drop(gt_df.columns[2], axis=1, inplace=True)
precision_array = []
recall_array = []

gt_grouped = gt_df.groupby(gt_df.columns[0])
gt_dict = {}

for name, group in gt_grouped:
    tmp = group.drop_duplicates()
    # if '__' in name:
    gt_dict[name] = tmp
num = 0
for name2, group2 in grouped:
    # if '__' in name2:
    try:
        tmp_df = group2.drop_duplicates()
        # if num == 1:
        #     print(tmp_df)
        #     print(gt_dict[name2])
        merged_df = pd.merge(tmp_df, gt_dict[name2], how='inner')
        # if num == 1:
        #     print(merged_df)
        # 计算合并后的行数
        same_rows_count = len(merged_df)  # 或者 merged_df.shape[0]

        precision = same_rows_count / len(tmp_df)
        recall = same_rows_count / len(gt_dict[name2])
        if num ==1:
            print(same_rows_count, precision, recall)
        num += 1

        # precision_1 += len(find_intersection)
        # sum_1 += len(result_set)
        precision_array.append(precision)
        recall_array.append(recall)
    except Exception as e:
        print(e)

sorted_precision = sorted(precision_array)
sorted_recall = sorted(recall_array)
print(len(sorted_precision))
sorted_precision = sorted_precision[300:]
print("Precision at k = ", k, "=", np.mean(sorted_precision))
print("Recall at k = ", k, "=", np.mean(recall_array))
# print("Precision at k = ", k, "=", len(precision_1), len(sum_1))
print("--------------------------")

print(sorted_precision[:30])
print(sorted_recall[:30])
