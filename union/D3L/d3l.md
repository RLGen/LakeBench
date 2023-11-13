<div>
    <h1>D3L</h1>
</div>

<br>

<h2>Folder Structure</h2>

```
.
├─── indexing.py                # Index the columns of tables
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

> -- root_path [Data path to be indexed, str]
>
> --name_file  --format_file --value_file --embdding_file --distriution_file [Created index file, str]

```sh
python indexing.py --root_path "/data/webtable/large/" --name_file "name.lsh" --format_file "format.lsh" --value_file "value.lsh" --embedding_file "embedding.lsh" --distribution_file "distribution.lsh"
```

**Step3: Querying**

> --output_folder [Path to store query results, str]
>
> --combined_file_path [Path to store final results, str]
>
> --K [top-*k*, int]
>
> --split_num [Number of processes in multiple processes, int]
>
> --name_index_file  --format_index_file --value_index_file --embdding_index_file --distriution_index_file [Created index file, str]

```sh
python query.py --output_folder "test" --combined_file_path "test.csv" --k 10 --split_num 10 --query_tables_folder "/data/webtable/small/" --name_index_file "name.lsh" --format_index_file "format.lsh" --value_index_file "value.lsh" --embedding_index_file "embedding.lsh" --distribution_index_file "distribution.lsh"
```

