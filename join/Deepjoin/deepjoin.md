<div align= "center">
    <h1> DeepJoin: Joinable Table Discovery with Pre-trained Language Models</h1>
</div>
<p align="center">
  <a href="#-community">Community</a> •
  <a href="#-struct">Folder Structure</a> •
  <a href="#-getstart">GettingStart</a> •
  <a href="#-quickstart">QuickStart</a> •
  <a href="#-result">Result</a> •
</p>




<br>

<div align="center">
<img src="img/deepjoin.png" width="1000px">
</div>
<br>

🌊  Our solution is an embeddingbased retrieval, which employs a pre-trained language model (PLM) and is designed as one framework serving both equi- and semantic (with a similarity condition on word embeddings) joins for textual attributes with fairly small cardinalities. We propose a set of contextualization options to transform column contents to a text sequence. The PLM reads the sequence and is fine-tuned to embed columns to vectors such that columns are expected to be joinable if they are close to each other in the vector space. Since the output of the PLM is fixed in length, the subsequent search procedure becomes independent of the column size. With a state-of-the-art approximate nearest neighbor search algorithm, the search time is sublinear in the repository size. To train the model, we devise the techniques for preparing training data as well as data augmentation. The experiments on real datasets demonstrate that by training on a small subset of a corpus, DeepJoin generalizes to large datasets and its precision consistently outperforms other approximate solutions'.DeepJoin is even more accurate than an exact solution to semantic joins when evaluated with labels from experts. Moreover, when equipped with a GPU, DeepJoin is up to two orders of magnitude faster than existing solutions
<span id="-community"></span>

## 👫 Community

We deeply appreciate the invaluable effort contributed by our dedicated team of developers, supportive users, and esteemed industry partners.

- [Massachusetts Institute of Technology](https://www.mit.edu/)
- [Beijing Institute of Technology](https://english.bit.edu.cn/)
- [Hong Kong University of Science and Technology](https://www.hkust-gz.edu.cn/)
- [Apache Flink](https://flink.apache.org/)
- [Intel](https://www.intel.com/)

<span id="-struct"></span>

## 📧 Folder Structure



```
.
├─── imgs                    # picture of different experiments
├─── join                    # join algorithms                
| ├─── Joise 
| ├─── LSH
| ├─── Pexeso         
| └─── DeepJoin         
| 
├─── union                   # union algorithms                
| ├─── TUS 
| ├─── D3L
| ├─── Santos         
| └─── Starmie  
| 
├─── join&union              # join&union algorithms               
| ├─── Joise 
| ├─── LSH
| ├─── Pexeso         
| └─── DeepJoin 
| 
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
conda create --name lakebench python=3.
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

The detailed instructions for downloading and processing are shown in <a href = "#-table_dataset">table</a> below. Please follow it to download datasets/queries before running or developing algorithms.

<div id="-table_dataset"></div> 

|                 Datasets                    | Queries | Ground Truth |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
|        [WebTable](https://drive.google.com/file/d/1tnI2EyrYHlc3fpv0SSMoe2sqWQZoOEjg/view?usp=drive_link)     |  [WebTable_Union_Query](https://drive.google.com/file/d/16VHNC8vvs3pE5ldrWyQzOKe6VkYVcIP8/view?usp=drive_link)     |  [WebTable_Union_Groun_Truth](https://drive.google.com/file/d/1P0au83zifpAn23xwFlDMV9azTBY-GgZz/view?usp=drive_link)     | 
|       [OpenData_SG](https://drive.google.com/file/d/1pPKMJ2Xnd6gYtkT_zVHIHCC97K5Yib4e/view?usp=drive_link)       |   [WebTable_Join_Query](https://drive.google.com/file/d/19YUPS2OsDQBmfR1ITtoKGvaIX4TO1pgg/view?usp=drive_link)       |    [WebTable_Join_Groun_Truth](https://drive.google.com/file/d/1wmh1qKRVGZ6zA4tEvwttnK9HdFpvbATs/view?usp=drive_link)       |  
|       [OpenData_CAN](https://drive.google.com/file/d/1ksOyaGVugeu7UJ0SKbYj4ri-rwgGfNH8/view?usp=drive_link)       |    [OpenData_Union_Query](https://drive.google.com/file/d/1SJ-SDOTkFyIfkHewVK-LDov3vrQdW9O4/view?usp=drive_link)       |  [OpenData_Union_Groun_Truth](https://drive.google.com/file/d/1_Bu5tw4Ou6yAO4DbE7x-wXUJ39pR2djv/view?usp=drive_link)       |  
|       [OpenData_UK、OpenData_USA](https://drive.google.com/drive/folders/1F9hIN815B6jmn85t-4gQGDoV-gLX8QvH?usp=drive_link)       |    [OpenData_Join_Query](https://drive.google.com/file/d/1em6YDiG311GfW2Inpxh7SomBA-zaLcmO/view?usp=drive_link)       |   [OpenData_Join_Groun_Truth](https://drive.google.com/file/d/16j_ZyHEFjYDiaTNSTbF8I-t1kNFtQWJ1/view?usp=drive_link)       |   

<span id="-quickstart"></span>

## 🐠 Quick Start

LakeBench is easy to use and extend. Going through the bellowing examples will help you familiar with LakeBench for quick use, evaluate an existing join/union algorithm on your own dataset, or developing new join/union algorithms.

### Train

Here is an example to train Starmie on *Webtable*. Training other supported algorithms (on other datasets with different query) can be specified by the <a href = "#-table">table</a> below.

**Step1: Check your environment**

You need to properly install nvidia driver first. To use GPU in a docker container You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)). Then, Please check your CUDA version via `nvidia-smi`

**Step2: Pretrain**

```sh
python pretrain.py --task webtable
```

**Step3: Indexing**

```sh
python index.py --benchmark webtable
```

**Step4: Querying**

```sh
python query.py --benchmark webtable --K 5 --N 10 --threshold 0.7
```

If you want to try other algorithms, you can read more details according to the table:

<div id="-table"></div> 

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



<span id="-result"></span>

<br>

##  🏆  Results



### Efficiency and Memory Usage Reference

Efficiency and Memory Usage of Table Join Search:

<div align="center">
<img src="imgs/table1.png" width="1000px">
</div>
<br>


Efficiency and Memory Usage of Table Union Search:

<div align="center">
<img src="imgs/table2.png" width="1000px">
</div>
