<div>
    <h1>Tabert</h1>
</div>


<br>

<h2>Folder Structure</h2>

```
.
├─── env.yml                 # environment of tabert
├─── index.py                # Index the columns of tables
├─── query.py                # Get the union results
├─── stept_env.sh
├─── hnsw_search.py     
└─── tabert.md
```

<br>

<h2>Training Steps</h2>

**Step1: Check your environment**

First, install the conda environment `tabert` with supporting libraries.

```bash
bash setup_env.sh
```

Once the conda environment is created, install `TaBERT` using the following command:

```bash
conda activate tabert
pip install --editable .
```

**Step 2: download Pre-trained Models**

The author provides four Pre-trained models. Pre-trained models could be downloaded from command line as follows:

```sh
pip install gdown

# TaBERT_Base_(k=1)
gdown 'https://drive.google.com/uc?id=1-pdtksj9RzC4yEqdrJQaZu4-dIEXZbM9'

# TaBERT_Base_(K=3)
gdown 'https://drive.google.com/uc?id=1NPxbGhwJF1uU9EC18YFsEZYE-IQR7ZLj'

# TaBERT_Large_(k=1)
gdown 'https://drive.google.com/uc?id=1eLJFUWnrJRo6QpROYWKXlbSOjRDDZ3yZ'

# TaBERT_Large_(K=3)
gdown 'https://drive.google.com/uc?id=17NTNIqxqYexAzaH_TgEfK42-KmjIRC-g'
```

**Step3: Indexing**

Here are some parameters:

> --benchmark [Choose benchmark, str] [opendata, opendata_large, webtable, webtable_large]
>
> --model_path [Path to Pre-trained Model, str]
>
> --file_type [Type of table, str] [.csv, .xlsx]

```sh
python index.py --benchmark webtable --model_path /tabert_base_k3/model.bin  --file_tye .csv
```

**Step4: Querying**

> --benchmark [Choose benchmark, str] [opendata, opendata_large, webtable, webtable_large]
>
> --K [top-$k$ ,int] [5~60]
>
> --threshold [Choose threshold, float] [0.5~0.9]
>
> --N [Choose N, int] [4, 10, 16, 25, 50]

```sh
python query.py --benchmark webtable --K 5 --N 10 --threshold 0.7
```

