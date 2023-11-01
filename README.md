<div align= "center">
    <h1> LakeBench: A Benchmark for Discovering Joinable and Unionable Tables in Data Lakes</h1>
</div>
<p align="center">
  <a href="#-community">Community</a> •
  <a href="#-leaderboard">leaderborad</a> •
  <a href="#-data-and-query">Data and Query</a> •
  <a href="#-getstart">GettingStart</a> •
  <a href="#-quickstart">QuickStart</a> •
  <a href="#-result-uploading">Result Uploading</a> •
</p>



<br>

<div align="center">
<img src="imgs/framework.png" width="1000px">
</div>
<br>

🌊  LakeBench is a a large-scale benchmark designed to test the mettle of *table discovery* methods on a much larger scale, providing a more comprehensive and realistic evaluation platform for the field, including *finance, retail, manufacturing, energy, media, and more.*

  Despite their paramount significance, existing benchmarks for evaluating and supporting *table discovery* processes have been limited in scale and diversity, often constrained by small dataset sizes. They are not sufficient to systematically evaluate the effectiveness, scalability, and efficiency of various solutions.

  LakeBench consists of over 16 million real tables **–1,600X** larger than existing data lakes, frommultiple sources, with an overall size larger than 1TB (**100X** larger). LakeBench contains both synthesized and real queries, in total more than 10 thousand queries –**10X** more than existing benchmarks, for join and union search respectively. 

🙌  With LakeBench, we thoroughly evaluate the state-of-the-art *table discovery* approaches on our benchmark and present our experimental findings from diverse perspectives, which we believe can push the research of *table discovery*.

<span id="-community"></span>

## 👫 Community

We deeply appreciate the invaluable effort contributed by our dedicated team of developers, supportive users, and esteemed industry partners.

- [Massachusetts Institute of Technology](https://www.mit.edu/)
- [Beijing Institute of Technology](https://english.bit.edu.cn/)
- [Hong Kong University of Science and Technology](https://www.hkust-gz.edu.cn/)
- [Apache Flink](https://flink.apache.org/)
- [Intel](https://www.intel.com/)

<span id="-leaderboard"></span>

## 🏆 Leaderboard

This leaderboard showcases the performance of various algorithms on LakeBench. Two performance metrics are adopted: *(i) Efficiency* : the time of both offline index building and online query processing ; *(ii)Memory Usage* : The memory consumption of both offline index building and online query processing.

<br>





<span id="-getstart"></span>

## 🐳 Getting Started

This is an example of how to set up LakeBench locally. To get a local copy up, running follow these simple example steps.

### Prerequisites

LakeBench is bulit on pytorch, with torchvision, torchaudio, and transfrmers.

To insall the required packages, you can create a conda environmennt:

```sh
conda create --name usb python=3.8
```

then use pip to install -r requirements.txt

```sh
pip install -r requirements.txt
```

From now on, you can start use LakeBench by typing 

```sh
python test.py
```

### Prepare Datasets

The detailed instructions for downloading and processing are shown in [Dataset Download](./preprocess/). Please follow it to download datasets before running or developing algorithms.

<span id="-quickstart"></span>

## 🐳  Quick Start

LakeBench is easy to use and extend. Going through the bellowing examples will help you familiar with LakeBench for quick use, evaluate an existing join/union algorithm on your own dataset, or developing new join/union algorithms.

### Train

Here is an example to train Starmie on Webtable with small query. Training other supported algorithms (on other datasets with different query) can be specified by a config file.

**Step1: Check your environment**

You need to properly install nvidia driver first. To use GPU in a docker container You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)). Then, Please check your CUDA version via `nvidia-smi`

**Step2: Pretrain**

```sh
python pretrain.py --c config/union/pretrain/starmie/webtable_small.yaml
```

**Step3: Indexing**

```sh
python index.py --c config/union/index/starmie/webtable_small.yaml
```

**Step4: Querying**

```sh
python query.py --c config/union/query/starmie/webtable_small.yaml
```

### Evalution

After training, you can check the evaluation performance on training logs, or running evaluation script:

```sh
python eval.py --datasets webtable_small --methods starmie --td union
```

##  📧 Result Uploading



### Efficiency and Memory Usage Reference

Efficiency and Memory Usage of Table Join Search:

<div align="center">
<img src="imgs/table1.png" width="1000px">
</div>



Efficiency and Memory Usage of Table Union Search:

<div align="center">
<img src="imgs/table2.png" width="1000px">
</div>

Note that for larger datasets like WebTable, please make sure enough memory is allocated. 
