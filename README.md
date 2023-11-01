<div align= "center">
    <h1> LakeBench: A Benchmark for Discovering Joinable and Unionable Tables in Data Lakes</h1>
</div>
<p align="center">
  <a href="#-community">Community</a> •
  <a href="#-leaderboard">leaderborad</a> •
  <a href="#-data-and-query">Data and Query</a> •
  <a href="#-getstart">GettingStart</a> •
  <a href="#-quickstart">QuickStart</a> •
  <a href="#-result">Result</a> •
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

## 🏆 Folder Structure



```
.
├─── config                  # setting parameters        
├─── pic                     # picture of different experiments
├─── model                   # save model
├─── Datasets                
| ├─── original              # original datasets
| └─── processing            # train_data/val_data/test_data
├─── RLGen
| ├─── AC.py                 # framework of RLGen
| ├─── Classifier.py         # choose Weak/Semi classification
| ├─── Env.py                # RL environment
| ├─── Processing_data.py    # get data information
| ├─── Test.py               # downstream_model:mlp/svm/LR/NB
| └─── main.py               
├─── GEN
| ├─── GAN.py                # codes of CTGAN
| ├─── Classifier.py         # choose Weak/Semi classification
| ├─── Test.py               # downstream_model:mlp/svm/LR/NB
| └─── main.py 
├─── LSTM
| ├─── LSTM.py               # codes of LSTM
| ├─── Classifier.py         # choose Weak/Semi classification
| ├─── Test.py               # downstream_model:mlp/svm/LR/NB
| └─── main.py               
├─── main.py                 # evaluation of different methods
├─── README.md
└─── requirements.txt
```

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

If you want to try other algorithms, here are other training steps:

|                 Algorithms                  |     Task     |                            Train                             |
| :-----------------------------------------: | :----------: | :----------------------------------------------------------: |
|        [Joise](join/Joise/joise.md)         |     Join     |         [./join/Joise/joise.md](join/Joise/joise.md)         |
|       [LSH Ensemble](join/LSH/lsh.md)       |     Join     |             [./join/LSH/lsh.md](join/LSH/lsh.md)             |
|       [Pexeso](join/Pexeso/pexeso.md)       |     Join     |       [./join/Pexeso/pexeso.md](join/Pexeso/pexeso.md)       |
|    [DeepJoin](join/Deepjoin/deepjoin.md)    |     Join     |   [./join/Deepjoin/deepjoin.md](join/Deepjoin/deepjoin.md)   |
|           [TUS](union/TUS/tus.md)           |    Union     |            [./union/TUS/tus.md](union/TUS/tus.md)            |
|           [D3L](union/D3L/d3l.md)           |    Union     |            [./union/D3L/d3l.md](union/D3L/d3l.md)            |
|      [Santos](union/Santos/santos.md)       |    Union     |      [./union/Santos/santos.md](union/Santos/santos.md)      |
|     [Starmie](union/Starmie/starmie.md)     |    Union     |    [./union/Starmie/starmie.md](union/Starmie/starmie.md)    |
|      [Frt12](join&union/Frt12/frt.md)       | Join & Union |     [./join&union/Frt12/frt.md](join&union/Frt12/frt.md)     |
| [InfoGather](join&union/InfoGather/info.md) | Join & Union | [./join&union/InfoGather/info.md](join&union/InfoGather/info.md) |
|     [Aurum](join&union/Aurum/aurum.md)      | Join & Union |   [./join&union/Aurum/aurum.md](join&union/Aurum/aurum.md)   |



### Evalution

After training, you can check the evaluation performance on training logs, or running evaluation script:

```sh
python eval.py --datasets webtable_small --methods starmie --td union
```

<span id="-result"></span>

##  📧 Results



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
