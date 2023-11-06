import pickle
import time

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

# result_list = ['nihao','tahap','hipopo']
length = len(result_list)
freq = {}
for i in range(length):
    for j in range(i + 1, length):
        mid_res1 = result_list[i]
        mid_res2 = result_list[j]
        mid = []
        count = 0
        for k in att:
            if mid_res1 in k and mid_res2 in k:
                count += 1
        mid.append(mid_res1)
        mid.append(mid_res2)
        # freq[mid] = count
        freq[tuple(mid)] = count
        # my_dict = {tuple(mid): count}
with open(file='two_att_freq.pkl',mode='wb') as f:
    pickle.dump(freq, f)

print(freq)
e = time.time()
print(e-s)