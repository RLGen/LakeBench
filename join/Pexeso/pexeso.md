<div>
    <h1>Pexeso</h1>
</div>


<br>

<h2>Folder Structure</h2>

```
.
├─── pretrain.py             # Pretrain
├─── index.py                # Index the columns of tables
├─── query.py                # Get the union results                         
├─── hnsw_search.py     
└─── pexeso.md
```

<br>

<h2>Quik Start</h2>

**Step1: Check your environment**

You need to properly install python package first. Please check package you have installed via `pip list`

**Step2: Pretrain**

```sh
#webtable
python pretrain.py --c config/pretrain/webtable/pexeso.yaml
#opendata
python pretrain.py --c config/pretrain/opendata/pexeso.yaml
```

**Step3: Indexing**

```sh
#webtable
python index.py --c config/index/webtable/pexeso.yaml
#opendata
python index.py --c config/index/opendata/pexeso.yaml
```

**Step4: Querying**

```sh
#webtable
python query.py --c config/query/webtable/pexeso.yaml
#opendata
python query.py --c config/query/opendata/pexeso.yaml
```