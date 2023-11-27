<h1 align="center">D3L Data Discovery Framework</h1>
<p align="center">Similarity-based data discovery in data lakes</p>

<p align="center">
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Getting started

This is an approximate implementation of the [D3L research paper](https://arxiv.org/pdf/2011.10427.pdf) published at ICDE 2020.
The implementation is approximate because not all notions proposed in the paper are transferred to code. The most notable differences are mentioned below:
* The indexing evidence for numerical data is different from the one presented in the paper. In this package, numerical columns are transformed to their density-based histograms and indexed under a random projection LSH index.
* The distance aggregation function (Equation 3 from the paper) is not yet implemented. In fact, the aggregation function is customizable. During testing, a simple average of distances has proven comparable to the level reported in the paper.
* The package uses similarity scores (between 0 and 1) instead of distances, as described in the paper.
* The join path discovery functionality from the paper is not yet implemented. This part of the implementation will follow shortly. 

## Installation

You'll need Python 3.6.x to use this package.

```
pip install git+https://github.com/alex-bogatu/d3l
```

### Installing from a specific release

You may wish to install a specific release. To do this, you can run:

```
pip install git+https://github.com/alex-bogatu/d3l@{tag|branch}
```

Substitute a specific branch name or tag in place of `{tag|branch}`.

## Usage

See [here](./examples/notebooks/D3L_hello_world.ipynb) for an example notebook.

However, keep in mind that this is a BETA version and future releases will follow. Until then, if you encounter any issues feel free to raise them [here](https://github.com/alex-bogatu/d3l/issues).


<h2>Instruction</h2>

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

