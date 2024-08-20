import math

total_number_of_sets = 1.0

def init():

    global total_number_of_sets
    total_number_of_sets = 1.0

def pruning_power_ub(freq, k):
    return math.log(((min(k, freq) + 0.5) * (total_number_of_sets - k - freq + min(k, freq) + 0.5)) /
                    (((max(0, k - freq) + 0.5) * (max(freq - k, 0) + 0.5))))

def inverseSetFrequency(freq):
    return math.log(total_number_of_sets / freq)

#判断重复多少，要跳到哪
def nextDistinctList(tokens, gids, curr_list_index):
    if curr_list_index == len(tokens) - 1:
        return len(tokens), 0
    num_skipped = 0
    for i in range(curr_list_index + 1, len(tokens)):
        if i < len(tokens) - 1 and gids[i + 1] == gids[i]:
            num_skipped += 1
            continue
        return i, num_skipped
    return len(tokens), 0

#用于计算两个已排序的 token 集合之间的重叠数量。
def overlap(set_tokens, query_tokens):
    i, j = 0, 0
    overlap = 0
    while i < len(query_tokens) and j < len(set_tokens):
        d = query_tokens[i] - set_tokens[j]
        if d == 0:
            overlap += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        elif d > 0:
            j += 1
    return overlap

# Helper functions
def overlap(list1, list2):
    res = [v for v in list1 if v in list2]
    # print(res)
    return len(res)
    #return len(set1.intersection(set2))

def overlap_and_update_counts(set_tokens, query_tokens, counts):
    i, j = 0, 0
    overlap = 0
    while i < len(query_tokens) and j < len(set_tokens):
        d = query_tokens[i] - set_tokens[j]
        if d == 0:
            counts[i] -= 1
            overlap += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        elif d > 0:
            j += 1
    return overlap
