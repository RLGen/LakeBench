<div>
    <h1>Joise</h1>
</div>


<h2>Quick Start</h2>

**Step1: offline**

#### parameters：
cpath: string,path of data lake
save_root: string, save path of the index file created in this step
#### modules:
createRawTokens.py: Read all the files in the data lake and store them in save_root/rawTokens.csv. Assign a setID to each candidate column and store the corresponding relationship in save_root/setMap.pkl
createIndex.py:Create an inverted index based on rawTokens.csv, and save the results in save_root/outputs/.


```sh
python offline/offline_api.py --cpath --save_root
```

**Step2: online**

```sh
#webtable
python index.py --c config/index/webtable/joise.yaml
#opendata
python index.py --c config/index/opendata/joise.yaml
```

**Step4: Querying**

#### parameters：
qpath: string,path of all query tables
save_root: string,path of index
result_root：string,path of query results
k:int, find top-k sets that have the largest intersection with query
#### modules:
cost.py: Functions that calculate  the cost of reading sets or posting lists.
heap.py: Operations related to the heap that stores the top k results.
josie.py: Joise algorithm
josie_util.py:Operations related to joise algorithm.
data_process.py:Data processing functions needed when creating the index.
common.py: Other functions

```sh
python online/online_api.py --cpath --save_root --result_root --k
```