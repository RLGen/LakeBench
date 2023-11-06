from common import nextDistinctList
from cost import readSetCost


#当前的candidate
class CandidateEntry:
    def __init__(self, id, size, candidateCurrentPosition, queryCurrentPosition, skippedOverlap):
        self.id = id
        self.size = size
        self.firstMatchPosition = candidateCurrentPosition  #Jx,0
        self.latestMatchPosition = candidateCurrentPosition  #Jx
        self.queryFirstMatchPosition = queryCurrentPosition   #i
        self.partialOverlap = skippedOverlap + 1   #目前重叠的tokens数量

        self.maximumOverlap = 0  #ub
        self.estimatedOverlap = 0   #est
        self.estimatedCost = 0.0   #the I/O time cost of reading this set
        self.estimatedNextUpperbound = 0
        self.estimatedNextTruncation = 0
        self.read = False

    # 第一次看到它时创建一个新条目 queryCurrentPosition 和 CandidateCurrentPosition 有一个匹配的标记
    @staticmethod
    def newCandidateEntry(id, size, candidateCurrentPosition, queryCurrentPosition, skippedOverlap):
        ce = CandidateEntry(id, size, candidateCurrentPosition, queryCurrentPosition, skippedOverlap)
        return ce

    #  当在 queryCurrentPosition 和 CandidateCurrentPosition 之间找到新的重叠token时更新
    def update(self, candidateCurrentPosition, skippedOverlap):
        self.latestMatchPosition = candidateCurrentPosition
        self.partialOverlap = self.partialOverlap + skippedOverlap + 1

    #计算ub 最大交叉数 公式6
    def upperboundOverlap(self, querySize, queryCurrentPosition):

        self.maximumOverlap = self.partialOverlap + min(querySize - queryCurrentPosition - 1, self.size - self.latestMatchPosition - 1)
        return self.maximumOverlap

    #计算est 公式4 estimatedOverlap
    def estOverlap(self, querySize, queryCurrentPosition):
        self.estimatedOverlap = int(float(self.partialOverlap) / float(queryCurrentPosition + 1 - self.queryFirstMatchPosition) * float(querySize - self.queryFirstMatchPosition))
        self.estimatedOverlap = min(self.estimatedOverlap, self.upperboundOverlap(querySize, queryCurrentPosition))
        return self.estimatedOverlap

    def estCost(self):
        self.estimatedCost = readSetCost(self.suffixLength())
        return self.estimatedCost

    def estTruncation(self, querySize, queryCurrentPosition, queryNextPosition):
        self.estimatedNextTruncation = int(float(queryNextPosition - queryCurrentPosition) / float(querySize - self.queryFirstMatchPosition) * float(self.size - self.firstMatchPosition))
        return self.estimatedNextTruncation

    # 公式9 pl的est
    def estNextOverlapUpperbound(self, querySize, queryCurrentPosition, queryNextPosition):
        queryJumpLength = queryNextPosition - queryCurrentPosition   #i'-i
        queryPrefixLength = queryCurrentPosition + 1 - self.queryFirstMatchPosition
        additionalOverlap = int(float(self.partialOverlap) / float(queryPrefixLength) * float(queryJumpLength))
        #j'xest 公式10
        nextLatestMatchingPosition = int(float(queryJumpLength) / float(querySize - self.queryFirstMatchPosition) * float(self.size - self.firstMatchPosition)) + self.latestMatchPosition
        #ub,est 公式11
        self.estimatedNextUpperbound = self.partialOverlap + additionalOverlap + min(querySize - queryNextPosition - 1, self.size - nextLatestMatchingPosition - 1)
        return self.estimatedNextUpperbound


    def suffixLength(self):
        return self.size - self.latestMatchPosition - 1

    def checkMinSampleSize(self, queryCurrentPosition, batchSize):
        return (queryCurrentPosition - self.queryFirstMatchPosition + 1) > batchSize


# Sorting wrappber for counter entry

class ByEstimatedOverlap:
    def __init__(self, candidateEntries):
        self.candidateEntries = candidateEntries

    def less(self, i,j):
        if self.candidateEntries[i].estimatedOverlap == self.candidateEntries[j].estimatedOverlap:
            return self.candidateEntries[i].estimatedCost < self.candidateEntries[j].estimatedCost
        return self.candidateEntries[i].estimatedOverlap > self.candidateEntries[j].estimatedOverlap

    def __len__(self):
        return len(self.candidateEntries)

    def Swap(self, i,j):
        swap=self.candidateEntries[i]
        self.candidateEntries[i]=self.candidateEntries[j]
        self.candidateEntries[j]=swap

# Sort by maximum overlap in increasing order
class ByMaximumOverlap:
    def __init__(self, candidateEntries):
        self.candidateEntries = candidateEntries

    def less(self, other):
        return self.candidateEntries[self.i].maximumOverlap < self.candidateEntries[other.i].maximumOverlap

    def __len__(self):
        return len(self.candidateEntries)

    def Swap(self, i, j):
        swap = self.candidateEntries[i]
        self.candidateEntries[i] = self.candidateEntries[j]
        self.candidateEntries[j] = swap

#Sort by future maximum overlap in increasing order
class ByFutureMaxOverlap:
    def __init__(self, candidateEntries):
        self.candidateEntries = candidateEntries

    def less(self, other):
        return self.candidateEntries[self.i].estimatedNextUpperbound < self.candidateEntries[other.i].estimatedNextUpperbound

    def __len__(self):
        return len(self.candidateEntries)

    def Swap(self, i, j):
        swap = self.candidateEntries[i]
        self.candidateEntries[i] = self.candidateEntries[j]
        self.candidateEntries[j] = swap

    # 计算当前发布列表位置处未见过的候选者的重叠上限
def upperboundOverlapUknownCandidate(querySize, queryCurrentPosition, prefixOverlap):

    return querySize - queryCurrentPosition + prefixOverlap


def nextBatchDistinctLists(tokens, gids, currIndex, batchSize):
    # Find the end index of the next batch of distinct lists
    n = 0
    nextIndex = nextDistinctList(tokens, gids, currIndex)[0]  #Defined in common.py
    while nextIndex < len(tokens):
        currIndex = nextIndex
        n += 1
        if n == batchSize:
            break
        nextIndex = nextDistinctList(tokens, gids, currIndex)[0]
    return currIndex

#前缀长度是我们要读取的发布列表的数量 p
def prefixLength(querySize, kthOverlap):
    if kthOverlap == 0:
        return querySize
    return querySize - kthOverlap + 1

# 读取一批pl的benefit 公式12
def readListsBenenfitForCandidate(ce, kthOverlap):
    # Estimate the benefit of reading additional lists for a candidate
    if kthOverlap >= ce.estimatedNextUpperbound:
        return ce.estimatedCost
    return ce.estimatedCost - readSetCost(ce.suffixLength() - ce.estimatedNextTruncation)
    #readSetCost Defined in cost.py


#   处理counter中未读的候选者以获得合格候选者的排序列表  并计算读取下一批pl的benefit。
def processCandidatesInit(querySize, queryCurrentPosition, nextBatchEndIndex, kthOverlap, minSampleSize, candidates, ignores):
    readListsBenefit = 0.0
    numWithBenefit = 0
    qualified = []
    to_be_removed = []  # 存储需要删除的元素

    for ce in candidates.values():
        # Compute upper bound overlap
        ce.upperboundOverlap(querySize, queryCurrentPosition)
        # 标记需要删除的candidate 上界小于kth
        if kthOverlap >= ce.maximumOverlap:
            to_be_removed.append(ce.id)
            ignores[ce.id] = True
            continue
        # Candidate does not qualify if the estimation std err is too high
        if not ce.checkMinSampleSize(queryCurrentPosition, minSampleSize):
            continue
        # Compute estimation
        ce.estCost()
        ce.estOverlap(querySize, queryCurrentPosition)
        ce.estTruncation(querySize, queryCurrentPosition, nextBatchEndIndex)
        ce.estNextOverlapUpperbound(querySize, queryCurrentPosition, nextBatchEndIndex)
        # Compute read list benefit
        readListsBenefit += readListsBenenfitForCandidate(ce, kthOverlap)
        # Add qualified candidate for reading
        qualified.append(ce)
        if ce.estimatedOverlap > kthOverlap:
            numWithBenefit += 1

    # Remove the disqualified candidates
    for id in to_be_removed:
        del candidates[id]

    return readListsBenefit, numWithBenefit, qualified


#请注意，上述代码中的candidates和ignores参数在Python中需要使用字典数据结构来表示。
# 在调用processCandidatesInit函数时，您需要相应地传递字典参数。



def processCandidatesUpdate(kthOverlap, candidates, counter, ignores):
    readListsBenefit = 0.0
    for j, ce in enumerate(candidates):
        if ce is None or ce.read:
            continue
        if ce.maximumOverlap <= kthOverlap:
            # Setting the entry to None marking it eliminated to the caller.
            candidates[j] = None
            del counter[ce.id]
            ignores[ce.id] = True
        # Compute read list benefit for qualified candidate.
        readListsBenefit += readListsBenenfitForCandidate(ce, kthOverlap)
    return readListsBenefit

#// 计算读取产生新的第 k 个重叠的候选集的好处
def readSetBenefit(querySize, kthOverlap, kthOverlapAfterPush, candidates, readListCosts, fast):
    b = 0.0
    if kthOverlapAfterPush <= kthOverlap:
        return b
    p0 = prefixLength(querySize, kthOverlap)
    p1 = prefixLength(querySize, kthOverlapAfterPush)
    b += readListCosts[p0-1] - readListCosts[p1-1]
    if fast:
        return b
    for ce in candidates:
        if ce is None or ce.read:
            continue
        if ce.maximumOverlap <= kthOverlapAfterPush:
            # Add benefit from eliminating the candidate.
            b += ce.estimatedCost
    return b
