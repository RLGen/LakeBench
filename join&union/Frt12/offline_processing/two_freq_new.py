import pickle
import time
from tqdm import tqdm

s = time.time()
att = []
with open('candidate_attributes_list.pkl', 'rb') as f:
    att = pickle.load(f)

# 使用集合来存储不重复的字符串

freq = {}
for att_item in tqdm(att):
    for i in range(len(att_item)):
        for j in range(i + 1, len(att_item)):
            mid1 = []
            mid1.append(att_item[i])
            mid1.append(att_item[j])
            mid1 = sorted(mid1)
            t1 = tuple(mid1)
            if t1 in freq:
                freq[t1] += 1
            else:
                freq[t1] = 1


with open(file = 'two_att_freq.pkl',mode = 'wb') as f:
    pickle.dump(freq, f)
e = time.time()
print(e-s)
