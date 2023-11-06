import pickle
import time
from multiprocessing import Process, Queue
import multiprocessing
from tqdm import tqdm

s = time.time()
att = []
with open('candidate_attributes_list.pkl', 'rb') as f:
    att = pickle.load(f)
print("att_len{}".format(len(att)))
freq = {}

def split_list(lst, num_parts):
    avg = len(lst) // num_parts
    remainder = len(lst) % num_parts

    result = []
    start = 0
    for i in range(num_parts):
        if i < remainder:
            end = start + avg + 1
        else:
            end = start + avg
        result.append(lst[start:end])
        start = end

    return result

def subprocess(subatt,queue,queue_freq):
    freq_tmp = {}
    for att_item in subatt:
        for i in range(len(att_item)):
            for j in range(i + 1, len(att_item)):
                mid1 = []
                mid1.append(att_item[i])
                mid1.append(att_item[j])
                mid1 = sorted(mid1)
                t1 = tuple(mid1)
                if t1 in freq:
                    freq_tmp[t1] += 1
                else:
                    freq_tmp[t1] = 1
        queue.put(1)
    queue_freq.put(freq_tmp)
    queue.put((-1, "test-pid"))


split_num = 72
sub_att = split_list(att, split_num)

print("sub_att_len{}".format(len(sub_att[0])))


process_list = []
queue_freq = multiprocessing.Manager().Queue()

#####
# 为每个进程创建一个队列
queues = [Queue() for i in range(split_num)]
# queue = Queue()
# 一个用于标识所有进程已结束的数组
finished = [False for i in range(split_num)]

# 为每个进程创建一个进度条
bars = [tqdm(total=len(sub_att[i]), desc=f"bar-{i}", position=i) for i in range(split_num)]
# bar = tqdm(total=len(file_ls[0]), desc=f"process-{i}")
# 用于保存每个进程的返回结果
results = [None for i in range(split_num)]

for i in range(split_num):
    process = Process(target=subprocess, args=(sub_att[i], queues[i], queue_freq, ))
    process_list.append(process)
    process.start()

while True:
    for i in range(split_num):
        queue = queues[i]
        bar = bars[i]
        try:
            # 从队列中获取数据
            # 这里需要用非阻塞的get_nowait或get(True)
            # 如果用get()，当某个进程在某一次处理的时候花费较长时间的话，会把后面的进程的进度条阻塞着
            # 一定要try捕捉错误，get_nowait读不到数据时会跑出错误
            res = queue.get_nowait()
            if isinstance(res, tuple) and res[0] == -1:
                # 某个进程已经处理完毕
                finished[i] = True
                results[i] = res[1]
                continue
            bar.update(res)
        except Exception as e:
            continue

            # 所有进程处理完毕
    if all(finished):
        break

for process in process_list:
    process.join()


while not queue_freq.empty():
    try:
        k = queue_freq.get_nowait()
        for key,value in k:
            if key in freq:
                freq[key] += value
            else:
                freq[key] = value
    except Exception as e:
        continue

# my_dict = {tuple(mid): count}
with open(file='two_att_freq.pkl',mode='wb') as f:
    pickle.dump(freq, f)

# print(freq)
e = time.time()
print(e-s)
