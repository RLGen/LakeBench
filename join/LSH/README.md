### 功能模块

##### LSH Ensemble\datasketch\lsh.py： 

MinHashLSH类的实现代码

##### LSH Ensemble\datasketch\minhash.py：

Minhash类的实现代码

##### LSH Ensemble\datasketch\storage.py：

LSHensemble的底层存储相关

##### LSH Ensemble\datasketch\lshensemble.py：

MinHashLSHEnsemble类的实现代码

**主要函数**

初始化：加载参数。输入——Containment阈值threshold、num_perm、num_part、m、weights

index：进行索引，只调用一次。输入——(key, minhash, size)的list所代表的候选集合

query：进行查询。输入——查询集的minhash、size

##### LSH Ensemble\datasketch\lshensemble_partition.py

计算最优分区的功能模块

##### LSH Ensemble\lshensemble_benchmark.py

该部分为程序入口，执行时带参数  --level test/lite 即可采用对应的参数集合，参数集合在代码中设置。



### 参数设置

threshold：Containment阈值 取值范围[0.0, 1.0]。

num_perm：Minhash中所使用的排列函数的个数。

num_part：LSH Ensemble分区的个数。

m：LSH Ensemble使用大约比相同数量的MinHash LSH多出m倍的内存空间。另外，m越大，精确度越高。

weights：在优化参数设定时，对fp和fn重要性的权衡考量。

### 输出文件及中间文件

index.pickle，query.pickle：中间文件，将输入的索引和查询数据预处理为(key, minhash, size)的形式存储。

lshensemble_benchmark_query_results.csv：输出文件，包含精度召回以及时间。

1.0-16-256-match.csv：输出文件，按给定格式展示可匹配的column对。