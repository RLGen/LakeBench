import numpy as np
import copy
import pdb

class Grid:

    def __init__(self, id, level, mlevel, o, l):
        self.id = id
        self.level = level
        self.mlevel = mlevel
        self.o = o #中点座标
        self.l = l #网格边长
        self.child = {} # id - grid
        self.vector = []
        self.vec_ids = []
        self.emb = []

    def get_size(self):
        return len(self.vector)

    def is_leaf(self):
        return self.mlevel == self.level

    def get_leaf(self):
        # pdb.set_trace()
        if self.is_leaf():
            return [self]
        leaves = []
        for ch in self.child.values():
            leaves.extend(ch.get_leaf())
        return leaves

class HierarchicalGrid:
    # tree = {} 

    # grids = []

    def __init__(self, base, n_dims, n_layers, o, a, l):
        cordo = []
        for _ in range(n_dims):
            cordo.append(o+l/2)
        self.root = Grid(-1, 0, n_layers, cordo, l)
        self.base = base
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.o = o #座标最小值
        self.a = a #叶子节点长度
        self.l = l #总长度

        # print("build hg with %d pivot, %d level. So all %d^(%d * %d) = %d grids" % (
        #     n_dims, n_layers, base, n_dims, n_layers, base ** (n_layers * n_dims)))
        # print("原点为%f，方格大小为%f" % (o, a))

    def add_vector(self, vector, vec_id, emb):
        # add one data
        
        now = self.root
        pre = 0
        parts = self.base**self.n_dims
        nowl = self.l
        vec = copy.deepcopy(vector)
        cord_now = copy.deepcopy(now.o)

        for i in range(self.n_layers):
            bins = []
            nowl /= self.base
            for j in range(self.n_dims):
                bins.append(int(vec[j] / nowl))
                vec[j] %= nowl
                cord_now[j] = vector[j]-vec[j]+nowl/2
            grid_id = self.parsing_grid_id(bins, pre, parts)
            pre = grid_id
            if grid_id in now.child:
                grid = now.child[grid_id]
            else:
                grid = Grid(grid_id, i+1, self.n_layers, cord_now, nowl)
                now.child[grid_id] = grid
            if grid.is_leaf():
                grid.vector.append(vector)
                grid.vec_ids.append(vec_id)
                grid.emb.append(emb)
            now = grid
            # print("add vector in grid %d" % grid_id)
        return now

    def parsing_grid_id(self, bins, pre, parts):
        decimal = 0
        power = 0
        for i in range(len(bins)):
            decimal += bins[i] * pow(self.base, power)
            power += 1
        return decimal+pre*parts

    # def is_grid_existed(self, grid_id):
    #     if grid_id in self.tree:
    #         return True
    #     else:
    #         return False
    

    # def find_grid(self, vector):
    #     # 输入向量 返回对应grid id
    #     bins = []
    #     for i in range(self.n_dims):
    #         bins.append(int(vector[i] / self.a))
    #     grid_id = self.parsing_grid_id(bins, self.base)
    #     return grid_id

    # def vector_grid_filtering(self, vector, grid):
    #     # 判断向量是否在某id网格下
    #     if self.find_grid(vector) != grid.id:
    #         return True
    #     else:
    #         return False

    # def grid_grid_filtering(self, grid1, grid2):
    #     # 判断网格 与self无关？
    #     if grid1.id != grid2.id:
    #         return False
    #     else:
    #         return True

    # def get_grid_vector(self, id):
    #     return self.tree[id]


def build_hierarchical_grid(n_dims, data, embs, n_layers, x_min, x_max, a):
    
    # 每一层一维上分的块数
    base = 2

    # 初始化分层网格的列表
    hierarchical_grid = HierarchicalGrid(base=base, n_dims=n_dims, n_layers=n_layers, o=x_min, a=a, l=x_max-x_min)

    id_to_grid = {}
    for i in range(len(data)):
        id_to_grid[i] = hierarchical_grid.add_vector(data[i], i, embs[i])
    
    
    # 返回分层网格的列表以及value_id到grid的映射
    return hierarchical_grid, id_to_grid
