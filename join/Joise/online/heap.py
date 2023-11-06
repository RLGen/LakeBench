import heapq
import pandas as pd

class SearchResult:
    def __init__(self, ID, Overlap):
        self.ID = ID
        self.overlap = Overlap

    def __lt__(self, other):
        return self.overlap < other.overlap

class SearchResultHeap:
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def push(self, x):
        heapq.heappush(self.heap, x)

    def pop(self):
        return heapq.heappop(self.heap)

    def getHeap(self):
        result=[]
        for x in self.heap:
            # result.append((x.ID,x.overlap))
            result.append(x.ID)
        return result

    def showHeap(self,setMap):
        # print(self.heap)
        for x in self.heap:
            print("setID: ",x.ID)
            tbname=setMap[x.ID+1]["table_name"]
            cname=setMap[x.ID+1]["column_name"]
            print("set name:",cname)
            print("table name:",tbname)
            print("overlap: ",x.overlap)

    def orderHeap(self):
        self.heap.sort(key=lambda x: x.overlap, reverse=True)



def kthOverlap(h, k):
    if len(h) < k:
        return 0
    return h[0].overlap

def pushCandidate(h, k, id, overlap):
    if len(h) == k:
        if h.heap[0].overlap >= overlap:
            return False
        heapq.heappop(h.heap)
    heapq.heappush(h.heap, SearchResult(id, overlap))
    h.orderHeap()
    return True

def orderedResults(h):
    r = []
    while len(h) > 0:
        r.append(heapq.heappop(h))
    return r

# 列表表示堆，通过heapq模块的函数来实现堆的操作.

def kthOverlapAfterPush(h, k, overlap):
    if len(h) < k - 1:
        return 0
    kth = h[0].overlap
    if overlap <= kth:
        return kth
    if k == 1:
        return overlap
    jth = h[1].overlap if k == 2 else min(h[1].overlap, h[2].overlap)
    return min(jth, overlap)

def copyHeap(h):
    h2 = SearchResultHeap()
    h2.heap = h.heap.copy()
    return h2

def orderedResults(heap):
    return sorted(heap, reverse=True)

