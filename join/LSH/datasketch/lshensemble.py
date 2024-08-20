from collections import deque, Counter
import struct

import numpy as np
from datasketch.storage import _random_name
from datasketch.lsh import integrate, MinHashLSH
from datasketch.lshensemble_partition import optimal_partitions

# 给定containment 由式（23）计算FP概率,xq是x/q的比率。
def _false_positive_probability(threshold, b, r, xq):
    _probability = lambda t : 1 - (1 - (t/(1 + xq - t))**float(r))**float(b)
    if xq >= threshold:
        a, err = integrate(_probability, 0.0, threshold)
        return a
    a, err = integrate(_probability, 0.0, xq)
    return a

# 由式（24）计算FN概率
def _false_negative_probability(threshold, b, r, xq):
    _probability = lambda t : 1 - (1 - (1 - (t/(1 + xq - t))**float(r))**float(b))
    if xq >= 1.0:
        a, err = integrate(_probability, threshold, 1.0)
        return a
    if xq >= threshold:
        a, err = integrate(_probability, threshold, xq)
        return a
    return 0.0

# 为使假阳性和假阴性概率的加权和达到最小，计算最佳参数
def _optimal_param(threshold, num_perm, max_r, xq, false_positive_weight,
        false_negative_weight):
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm+1):
        for r in range(1, max_r+1):
            if b*r > num_perm:
                continue
            fp = _false_positive_probability(threshold, b, r, xq)
            fn = _false_negative_probability(threshold, b, r, xq)
            error = fp*false_positive_weight + fn*false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSHEnsemble(object):
    '''
    `E. Zhu et al. <http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf>`_.

    Args:
        threshold (float): Containment阈值 取值范围[0.0, 1.0]。
        num_perm (int, optional): Minhash中所使用的排列函数的个数。 
        num_part (int, optional): LSH Ensemble分区的个数。
        m (int, optional): 特殊超参数: LSH Ensemble使用大约比相同数量的MinHash LSH多出m倍的内存空间。
            另外，m越大，精确度越高。
        weights (tuple, optional): 在优化参数设定时，对fp和fn重要性的权衡考量。
        storage_config (dict, optional): 存储相关参数，不重要
        prepickle (bool, optional): 默认值由`storage_config`确定，不重要.

    Note:
        更多的分区(`num_part`)可以取得更好的准确性。
    '''

    def __init__(self, threshold=0.9, num_perm=128, num_part=16, m=8,
            weights=(0.5,0.5), storage_config=None, prepickle=None):
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if num_part < 1:
            raise ValueError("num_part must be at least 1")
        if m < 2 or m > num_perm:
            raise ValueError("m must be in the range of [2, num_perm]")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.threshold = threshold
        self.h = num_perm
        self.m = m
        rs = self._init_optimal_params(weights)
        # 对于r的每个可能取值，对LSHEnsemble中每个分区初始化一个LSH，索引初始化完成
        storage_config = {'type': 'dict'} if not storage_config else storage_config
        basename = storage_config.get('basename', _random_name(11))
        self.indexes = [
            dict((r, MinHashLSH(
                num_perm=self.h,
                params=(int(self.h/r), r),
                # 不同的分区存储方式可能不同
                storage_config=self._get_storage_config(
                    basename, storage_config, partition, r),
                prepickle=prepickle)) for r in rs)
            for partition in range(0, num_part)]
        self.lowers = [None for _ in self.indexes]
        self.uppers = [None for _ in self.indexes]

    # 给出一系列可能存在的xq，预处理得到使fp和fn带权和最优的参数b与r
    def _init_optimal_params(self, weights):
        false_positive_weight, false_negative_weight = weights
        self.xqs = np.exp(np.linspace(-5, 5, 10))
        self.params = np.array([_optimal_param(self.threshold, self.h, self.m,
                                               xq,
                                               false_positive_weight,
                                               false_negative_weight)
                                for xq in self.xqs], dtype=int)
        # 获取所有不同的r
        rs = set()
        for _, r in self.params:
            rs.add(r)
        return rs

    # 利用已有序列中获取预先处理的参数
    def _get_optimal_param(self, x, q):
        # xq序列中第一个大于等于实际x/q的下标
        i = np.searchsorted(self.xqs, float(x)/float(q), side='left')
        if i == len(self.params):
            i = i - 1
        return self.params[i]


    def _get_storage_config(self, basename, base_config, partition, r):
        config = dict(base_config)
        config["basename"] = b"-".join(
            [basename, struct.pack('>H', partition), struct.pack('>H', r)])
        return config

    def count_partition(self, data_sizes):
        # 创建最优的分区
        sizes, counts = np.array(sorted(
            Counter(data_sizes).most_common())).T
        partitions = optimal_partitions(sizes, counts, len(self.indexes))
        for i, (lower, upper) in enumerate(partitions):
            self.lowers[i], self.uppers[i] = lower, upper
        return partitions

    def index(self, entries):
        '''
        给出一个集合的键、MinHashes和大小的索引。

        Args:
            entries (`iterable` of `tuple`): 需要格式为 `(key, minhash, size)`，
                其中key是一个集合的唯一标识符，minhash是该集合的MinHash，size是该集合的大小。

        Note:
            size需要为正数
        '''
        # 将候选索引插入到对应分区
        for curr_part, (lower, upper) in enumerate(zip(self.lowers, self.uppers)):
            if entries[2] <= upper and entries[2] >= lower:
                for r in self.indexes[curr_part]:
                    self.indexes[curr_part][r].insert(entries[0], entries[1])
                break

    def index_batch(self, entries):
        '''
        给出一组集合的键、MinHashes和大小的索引。

        Args:
            entries (`iterable` of `tuple`): 需要格式为 `(key, minhash, size)`，
                其中key是一个集合的唯一标识符，minhash是该集合的MinHash，size是该集合的大小。

        Note:
            size需要为正数
        '''
        # 将候选索引插入到对应分区
        entries.sort(key=lambda e : e[2])
        curr_part = 0
        for key, minhash, size in entries:
            if size > self.uppers[curr_part]:
                curr_part += 1
            for r in self.indexes[curr_part]:
                self.indexes[curr_part][r].insert(key, minhash)

    def union(self, lshE):
        '''
        把另一lshensemble的值合并过来

        Args:
            lshE 待加的lsh索引
        '''
        for part in range(len(self.indexes)):
            for r in self.indexes[part]:
                for i in range(len(lshE.indexes[part][r].hashtables)):
                    for j in lshE.indexes[part][r].hashtables[i].keys():
                        # pdb.set_trace()
                        self.indexes[part][r].hashtables[i].insert(j, lshE.indexes[part][r].hashtables[i][j])
    
    def index_old(self, entries):
        '''
        给出所有集合的键、MinHashes和大小的索引。
        它只能在索引创建后被调用一次。

        Args:
            entries (`iterable` of `tuple`): 需要格式为 `(key, minhash, size)`，
                其中key是一个集合的唯一标识符，minhash是该集合的MinHash，size是该集合的大小。

        Note:
            size需要为正数
        '''
        if not self.is_empty():
            raise ValueError("Cannot call index again on a non-empty index")
        if not isinstance(entries, list):
            queue = deque([])
            for key, minhash, size in entries:
                if size <= 0:
                    raise ValueError("Set size must be positive")
                queue.append((key, minhash, size))
            entries = list(queue)
        if len(entries) == 0:
            raise ValueError("entries is empty")
        # 创建最优的分区
        sizes, counts = np.array(sorted(
            Counter(e[2] for e in entries).most_common())).T
        partitions = optimal_partitions(sizes, counts, len(self.indexes))
        for i, (lower, upper) in enumerate(partitions):
            self.lowers[i], self.uppers[i] = lower, upper
        # 将候选索引插入到对应分区
        entries.sort(key=lambda e : e[2])
        curr_part = 0
        for key, minhash, size in entries:
            if size > self.uppers[curr_part]:
                curr_part += 1
            for r in self.indexes[curr_part]:
                self.indexes[curr_part][r].insert(key, minhash)

    def query(self, minhash, size):
        '''
        给出查询集的MinHash和大小，检索出与查询集containment大于阈值的查询集。

        Args:
            minhash (datasketch.MinHash): 查询集合的minhash.
            size (int): 查询集的大小（unique value）.

        Returns:
            满足条件候选集的键值.
        '''
        for i, index in enumerate(self.indexes):
            u = self.uppers[i]
            if u is None:
                continue
            b, r = self._get_optimal_param(u, size)
            for key in index[r]._query_b(minhash, b):
                yield key

    # index中是否存在键值key
    def __contains__(self, key):
        return any(any(key in index[r] for r in index)
                   for index in self.indexes)

    # index是否为空
    def is_empty(self):
        return all(all(index[r].is_empty() for r in index)
                   for index in self.indexes)


if __name__ == "__main__":
    import numpy as np
    xqs = np.exp(np.linspace(-5, 5, 10))
    threshold = 0.5
    max_r = 8
    num_perm = 256
    false_negative_weight, false_positive_weight = 0.5, 0.5
    for xq in xqs:
        b, r = _optimal_param(threshold, num_perm, max_r, xq,
                false_positive_weight, false_negative_weight)
        print("threshold: %.2f, xq: %.3f, b: %d, r: %d" % (threshold, xq, b, r))