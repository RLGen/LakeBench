import pickle
import time
from tqdm import tqdm

s = time.time()
att = []
with open('candidate_attributes_list.pkl', 'rb') as f:
    att = pickle.load(f)

# 使用集合来存储不重复的字符串
unique_strings = set()
# 遍历嵌套列表，将每个内部列表中的字符串添加到集合中
for sublist in att:
    unique_strings.update(sublist)

# 将集合转换回列表
result_list = list(unique_strings)
# 打印结果列表

freq = {}

for i in tqdm(result_list):
    count = 0
    for item in att:
        if i in item:
            count += 1
    freq[i] = count
with open(file = 'one_freq_test.pkl',mode = 'wb') as f:
    pickle.dump(freq, f)

e = time.time()
print(e-s)
