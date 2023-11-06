import numpy
import pdb
import time

from Metric import dis_e_matrix, dis_e_point

# filter7_T = 

class Pair:
    # vectors = []

    def __init__(self, query, q_emb, clist, mlist):
        # q：查询向量 q_id：查询对应id clist：相关网格列表 
        self.q = query
        self.q_emb = q_emb
        self.clist = clist
        self.mlist = mlist

    def addm(self, grid):
        self.mlist.append(grid)
    
    def addc(self, grid):
        self.clist.append(grid)

    # def get_size(self):
    #     return len(self.vectors)

def addPair(Pairs, q, q_id, q_emb, grid, addType):
    if q_id not in Pairs.keys():
        Pairs[q_id] = Pair(q, q_emb, [], [])
    if addType=='M':
        Pairs[q_id].mlist.append(grid)
    elif addType=='C':
        Pairs[q_id].clist.append(grid)

def addPairs(Pairs, q, q_id, q_emb, grids, addType):
    if q_id not in Pairs.keys():
        Pairs[q_id] = Pair(q, q_emb, [], [])
    if addType=='M':
        Pairs[q_id].mlist.extend(grids)
    elif addType=='C':
        Pairs[q_id].clist.extend(grids)

# filter返回ture表示被筛掉 match返回true表示可匹配
def filter7(col, T, mismatch_map, qlen):
    if qlen-mismatch_map[col]<T:
        return True
    return False

def filter1(q, v, tao):
    n_dims = len(q)
    for i in range(n_dims):
        if q[i]-tao>v[i] or q[i]+tao<v[i]:
            return True
    return False

def match2(q, v, tao):
    n_dims = len(q)
    for i in range(n_dims):
        if q[i]+v[i]<=tao:
            return True
    return False

def filter3(q, c, tao):
    n_dims = len(q)
    for i in range(n_dims):
        if(abs(c.o[i]-q[i])>c.l/2+tao):
            return True
    return False

def filter4(cq, c, tao):
    n_dims = len(cq.o)
    for i in range(n_dims):
        if(abs(c.o[i]-cq.o[i])>c.l/2+cq.l/2+tao):
            return True
    return False

def match5(q, c, tao):
    n_dims = len(q)
    for i in range(n_dims):
        if(tao-q[i]>=c.o[i]+c.l/2):
            return True
    return False

def match6(cq, c, tao):
    n_dims = len(cq.o)
    for i in range(n_dims):
        # q_this_dim = [vec[i] for vec in cq.vector]
        # if(tao-min(q_this_dim)>=c.o[i]+c.l/2):
        if(tao-(cq.o[i]+cq.l/2)>=c.o[i]+c.l/2):
            return True
    return False

def block(Cq, Cr, Pairs, tao):
    for cq in Cq.child.values():
        for cr in Cr.child.values():
            if cq.is_leaf() and cr.is_leaf():
                for i in range(len(cq.vector)):
                    if match5(cq.vector[i], cr, tao):
                        addPair(Pairs, cq.vector[i], cq.vec_ids[i], cq.emb[i], cr, "M")
                    else:
                        if not filter3(cq.vector[i], cr, tao):
                            addPair(Pairs, cq.vector[i], cq.vec_ids[i], cq.emb[i], cr, "C")
            else:
                if match6(cq, cr, tao):
                    clist = cr.get_leaf()
                    for g in cq.get_leaf():
                        for i in range(len(g.vector)):
                            addPairs(Pairs, g.vector[i], g.vec_ids[i], g.emb[i], clist, "M")
                else:
                    if not filter4(cq, cr, tao):
                        block(cq, cr, Pairs, tao)

def verify(Pairs, I, tao, T, index_embs, query_embs, index_sets, match_map, mismatch_map, time_threshold):
    start = time.perf_counter()
    cnt1 = cnt2 = 0 ##
    over_time = False
    qlen = len(query_embs)
    for (q_id, pair) in Pairs.items():
        # match_col = {}
        for grid in pair.mlist:
            for col in I.search(grid):
                match_map[col] += 1
                # match_col[col] = 1
    for (q_id, pair) in Pairs.items():
        if over_time:
            break
        for grid in pair.clist: # 叶子网格
            if time.perf_counter() - start>time_threshold:
                over_time = True
                break
            for col in I.search(grid): #相关列
                if time.perf_counter() - start>time_threshold:
                    over_time = True
                    break
                if filter7(col, T*2, mismatch_map, qlen):
                    #  or match_col.get(col, 0)==1:
                    continue
                else:
                    for i in range(len(grid.vector)):#maybe can improve
                        if time.perf_counter() - start>time_threshold:
                            over_time = True
                            break
                        if grid.vec_ids[i] not in index_sets[col]:
                            continue
                        if match_map[col] >= T:
                        #  or filter7(col, T, mismatch_map, qlen):
                            #  or match_col.get(col, 0)==1:
                            break
                        if filter1(grid.vector[i], pair.q, tao):
                            mismatch_map[col] += 1
                            cnt1+=1
                        else:
                            if match2(grid.vector[i], pair.q, tao):
                                match_map[col] += 1
                                # match_col[col] = 1
                            else:
                                if dis_e_point(grid.emb[i], pair.q_emb) <= tao:
                                    match_map[col] += 1
                                    # match_col[col] = 1
                                else:
                                    mismatch_map[col] += 1
                                    cnt2+=1
                                    
    # for pair in mPair:
    #     for c in pair.clist:
    #         for col in I.search(c):
    #             match_map[col] += 1
    
    # for pair in cPair:
    #     for c in pair.clist: # 叶子网格
    #         for col in I.search(c): #相关列
    #             if filter7(col, T, mismatch_map, qlen):
    #                 continue
    #             else:
    #                 for i, vec in enumerate(c.vector):
    #                     if c.vec_ids[i] not in index_sets[col]:
    #                         continue
    #                     if match_map[col] >= T:
    #                         break
    #                     if filter1(vec, pair.q, tao):
    #                         mismatch_map[col] += 1
    #                     else:
    #                         if match2(vec, pair.q, tao):
    #                             match_map[col] += 1
    #                         else:
    #                             # pdb.set_trace()
    #                             if dis_e_point(index_embs[c.vec_ids[i]], query_embs[pair.q_id]) <= tao:
    #                                 match_map[col] += 1
    #                             else:
    #                                 mismatch_map[col] += 1

    res = []
    # pdb.set_trace()
    for key, val in match_map.items():
        if val>=T:
            res.append(key)
    
    if over_time:
        print("verify time over")
    else:
        print("verify time = {}".format(time.perf_counter() - start))

    return res,over_time
