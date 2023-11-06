import pickle
import time
from tqdm import tqdm

s = time.time()
att = []
with open('candidate_attributes_list.pkl', 'rb') as f:
    att = pickle.load(f)
#att = [['nihao','nini','byebbye'],['aaa', 'bbb','cccc','nini','byebbye']]
# 使用集合来存储不重复的字符串

freq = {}
for att_item in tqdm(att):
    for item in att_item:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1

with open(file = 'one_att_freq.pkl',mode = 'wb') as f:
    pickle.dump(freq, f)

print(freq)
e = time.time()
print(e-s)
