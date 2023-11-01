<div>
    <h1>D3L</h1>
</div>

<br>

<h2>Folder Structure</h2>

```
.
├─── index.py                # Index the columns of tables
├─── query.py                # Get the union results
├─── find.py 
├─── merge.py
├─── search.py     
└─── d3l.md
```

<br>

<h2>Training Steps</h2>

**Step1: Check your environment**

You need to properly install nvidia driver first. To use GPU in a docker container You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)). Then, Please check your CUDA version via `nvidia-smi`

**Step2: Indexing**

```sh
#webtable
python index.py --c config/index/webtable/d3l.yaml
#opendata
python index.py --c config/index/opendata/d3l.yaml
```

**Step3: Querying**

```sh
#webtable
python query.py --c config/query/webtable/d3l.yaml
#opendata
python query.py --c config/query/opendata/d3l.yaml
```

