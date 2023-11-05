from datasketch import MinHash, MinHashLSH

# 创建MinHashLSH对象
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# 构建MinHash索引
sets = [
    {'apple', 'banana', 'orange'},
    {'banana', 'pineapple', 'mango'},
    {'apple', 'grape', 'watermelon'},
    {'orange', 'kiwi', 'papaya'},
    {'apple', 'pear', 'orange'}
]

for i, s in enumerate(sets):
    # 创建MinHash对象
    minhash = MinHash(num_perm=128)
    
    # 添加集合元素到MinHash对象中
    for element in s:
        minhash.update(element.encode('utf-8'))
    
    # 添加MinHash对象到LSH索引中
    lsh.insert(i, minhash)

# 查询相似集合
query_set = {'apple', 'pear', 'orange'}
minhash_query = MinHash(num_perm=128)
for element in query_set:
    minhash_query.update(element.encode('utf-8'))

# 在LSH索引中查询相似集合,根据阈值来选择的
result = lsh.query(minhash_query)

print('相似集合：')
for r in result:
    print(sets[int(r)])
