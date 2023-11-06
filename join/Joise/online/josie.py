import time
import sys
from josie_util import *
from heap import *
from cost import *
from common import *
from common import overlap
from tqdm import tqdm
from dataProcess import *

batchSize = 5

'''在读取候选集和读取下一批发布列表之间进行选择时进行昂贵估计的预算。
是 num_candidate * num_estimation 的上限。
设置为0 强制快速估计所有candidate。 
设置为最大 int 数会强制进行昂贵的估计'''
#expensiveEstimationBudget=5000
expensiveEstimationBudget = sys.maxsize
# expensiveEstimationBudget = 0

#josie算法  integerSet, PLs, raw_tokens,rawDict,setMap, k, ignore,query_ID
def searchMergeProbeCostModelGreedy(integerSet,PLs,query,rawDict,setMap,k,ignoreSelf,queryID):
    # print('--进入josie--')
    tokens, freqs, gids = processQuery(query,rawDict)
    # print("实际token数量",len(tokens))
    # print(gids)
    if len(tokens)==0:
        return []
    
    readListCosts = [0] * len(freqs)

    for i in range(len(freqs)):
        if i == 0:
            readListCosts[i] = readListCost(freqs[i] + 1)
        else:
            readListCosts[i] = readListCosts[i - 1] + readListCost(freqs[i] + 1)
    # print(readListCosts)
    querySize = len(tokens)
    counter = {}   #存放candidateEntry, posting list里读入的
    ignores = {}   #bool
    if ignoreSelf:
        ignores[queryID] = True
    heap = SearchResultHeap()
    h=heap.heap
    currBatchLists = batchSize
    p=querySize
    for i in range(querySize):
        # print('*************************token第几个:', i + 1)
        #前缀过滤器
        if i+1>=p:
            break
        token = tokens[i] #
        # print(str(type(token)))
        numSkipped = nextDistinctList(tokens, gids, i)
        # print(numSkipped)
        skippedOverlap = numSkipped[0]
        maxOverlapUnseenCandidate = upperboundOverlapUknownCandidate(querySize, i, skippedOverlap)
        # print(maxOverlapUnseenCandidate)
        # 当达到当前位置的叠上限且counter中没有剩余集合，则提前终止
        if kthOverlap(h, k) >= maxOverlapUnseenCandidate and len(counter) == 0:
            # print("当达到当前位置的叠上限且counter中没有剩余集合，则提前终止")
            break
        #读取token的发布列表
        entries = getEntries(token,PLs)
        # print("entry长度",len(entries))
        #生成counter 跳过先前已计算出overlap的集合
        for entry in entries:
            # print(entry.ID)
            if entry.ID in ignores:
                # print("1")
                continue
            #已经见过的candidate
            if entry.ID in counter:
                ce = counter[entry.ID]
                ce.matchPosition = entry.MatchPosition
                ce.partialOverlap = skippedOverlap
                # print("2")
                continue
            #超了
            if kthOverlap(h, k) >= maxOverlapUnseenCandidate:
                # print("3")
                continue
            #加入新的candidate
            # print("4")
            counter[entry.ID] = CandidateEntry.newCandidateEntry(entry.ID, entry.Size, entry.MatchPosition,i, skippedOverlap)
        # print("len(counter)",len(counter))

        #当我们在最后一个列表里时终止，不需要再读取 set
        if i == querySize - 1:
            # print("#当我们在最后一个列表里时终止，不需要再读取 set")
            break

        # 当没有候选者时继续读取下一个列表, 在看到至少k个候选者之前，不要开始读取集合
        if len(counter) == 0 \
                or (len(counter) < k and len(h) < k) \
                or currBatchLists > 0:  #当我们仍在当前批次中时继续读取下一个列表

            currBatchLists -= 1
            # print("#当我们仍在当前批次中时继续读取下一个列表")
            continue

        #重置currBatchLists
        currBatchLists = batchSize
        #查找下一批发布列表的结束索引
        nextBatchEndIndex = nextBatchDistinctLists(tokens, gids, i, batchSize)
        #计算读取下一批发布列表的成本
        mergeListsCost = readListCosts[nextBatchEndIndex] - readListCosts[i]

        #处理candidate以估计阅读下一批posting list 的好处并获得合格的候选人
        #candidates：u 存放unread candidates列表 ，每一个元素对应一个candidateEntry
        mergeListsBenefit, numWithBenefit, candidates = processCandidatesInit(
            querySize, i+1, nextBatchEndIndex, kthOverlap(h, k), batchSize, counter, ignores)
        # print("candidates长度：",len(candidates))
        # 如果没有找到合格的候选人 或没有候选人可以带来benefit ->继续阅读postincs list
        if numWithBenefit == 0 or len(candidates) == 0:
            continue

        #candidates按照est从高到低排序
        candidates.sort(key=lambda c: c.estimatedOverlap, reverse=True)

        #跟踪估计预算
        prevKthOverlap = kthOverlap(h, k)
        numCandidateExpensive = 0
        fastEstimate = False
        fastEstimateKthOverlap = 0   #the kth overlap used for fast est.

        ii=0
        #贪婪地确定下一个最佳候选，直到1.找到合格的候选 2.候选耗尽 3.读取下一批list时产生更好的benifit
        for candidate in candidates:
            ii+=1
            if candidate is None:
                continue
            #|Q∩Uk|
            kth = kthOverlap(h, k)
            # print("candidate.estimatedOverlap,kth")
            # print(candidate.estimatedOverlap,kth)
            # 当当前候选人的est小于阈值时停止。
            if candidate.estimatedOverlap <= kth:
                # print("当当前候选人的est小于阈值时停止")
                # print("candidates长度："+str(ii))
                break
            # if ii>10000:
            #     print("taichangl")
            #     return 0
            # print("len(heap),k")
            # print(len(heap),k)
            # 当heap还没满时，总是读取candidate
            if len(heap) >= k:
                # 增加的候选人数量
                numCandidateExpensive += 1
                #如果达到估算预算，则切换到快速估算
                if not fastEstimate and numCandidateExpensive * len(candidates) > expensiveEstimationBudget:
                    fastEstimate = True
                    fastEstimateKthOverlap = prevKthOverlap
                #估计阅读下一批列表的好处
                if not fastEstimate:
                    mergeListsBenefit = processCandidatesUpdate(kth, candidates, counter, ignores)
                #估计阅读这个set的好处
                probeSetBenefit = readSetBenefit(querySize, kth, kthOverlapAfterPush(h, k, candidate.estimatedOverlap),
                                                 candidates, readListCosts, fastEstimate)
                probeSetCost = candidate.estimatedCost

                # 如果当前最好的候选人并不比阅读下一批发布列表更好，则停止寻找候选人
                # 下一个最好的候选人要么具有较低的效益，是单调的重叠或更高的成本。
                # 因此，如果当前最好的candidate没有更好，那么下一个最好的candidate将会更差。
                if probeSetBenefit - probeSetCost < mergeListsBenefit - mergeListsCost:
                    # print("如果当前最好的候选人并不比阅读下一批发布列表更好，则停止寻找候选人")
                    break

            #现在阅读这个candidate
            #如果我们使用快速估计，则减少阅读list的benefit
            if fastEstimate or (numCandidateExpensive + 1) * len(candidates) > expensiveEstimationBudget:
                mergeListsBenefit -= readListsBenenfitForCandidate(candidate, fastEstimateKthOverlap)

            #标记为已读
            candidate.read = True
            #再遇到就忽略
            ignores[candidate.id] = True

            del counter[candidate.id]
            # print("查看candidate",candidate.id)
            #如果这个候选还是小于K阈值，直接跳过。这有时在使用快速估计时可能会发生。
            if candidate.maximumOverlap <= kth:
                # print(candidate.maximumOverlap)
                # print("如果这个候选还是小于K阈值，直接跳过。这有时在使用快速估计时可能会发生。")
                continue

            #计算新的overlap
            if candidate.suffixLength() > 0:
                s = setTokensSuffix(integerSet, candidate.id, candidate.latestMatchPosition + 1)
                # print(str(type(s[0])),str(type(tokens[i+1])))
                suffixOverlap = overlap(s, tokens[i + 1:])
                # print("partialOverlap:",candidate.partialOverlap)
                totalOverlap = suffixOverlap + candidate.partialOverlap
            else:
                totalOverlap = candidate.partialOverlap

            #保存当前的k阈值
            prevKthOverlap = kth
            p=prefixLength(querySize,prevKthOverlap)
            # print("p:",p)
            #放入heap
            # heap.showHeap(setMap)
            # print("放入heap:",candidate.id,totalOverlap)
            pushCandidate(heap, k, candidate.id, totalOverlap)
            # heap.showHeap(setMap)
            result=heap.getHeap()
            # print("nowheaplen:",len(result))

    # print("还剩",len(counter.values()))
    #处理计数器中通过合并所有列表计算出的具有完全重叠的剩余集合
    for ce in counter.values():
        pushCandidate(heap, k, ce.id, ce.partialOverlap)
    result=heap.getHeap()

    # heap.showHeap(setMap)
    # print(result)
    # print('--出来josie--')
    return result
