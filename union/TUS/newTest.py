from simhash import Simhash, SimhashIndex

# 创建Simhash对象
sh1 = Simhash('This is a test string')
sh2 = Simhash('This is another test string')

# 创建SimhashIndex对象
index = SimhashIndex([], k=16)

# 添加Simhash对象到索引
index.add('sh1', sh1)
index.add('sh2', sh2)

# 查询与sh1相似的Simhash对象
result = index.get_near_dups(sh1)

print("Candidates with Hamming distance <= 3", result)