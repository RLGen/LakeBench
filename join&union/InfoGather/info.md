






<div align= "center">
    <h1> Paper：InfoGather- Entity Augmentation and Attribute Discovery By Holistic Matching with Web Tables   code for join and union</h1>
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
<img src="img/.jpg" width="1000px">
</div>
<br>

🌊  The Web contains a vast corpus of HTML tables, specifically entityattribute tables. We present three core operations, namely entity augmentation by attribute name, entity augmentation by example and attribute discovery, that are useful for "information gathering" tasks (e.g., researching for products or stocks). We propose to use web table corpus to perform them automatically. We require the operations to have high precision and coverage, have fast (ideally interactive) response times and be applicable to any arbitrary domain of entities. The naive approach that attempts to directly match the user input with the web tables suffers from poor precision and coverage. Our key insight is that we can achieve much higher precision and coverage by considering indirectly matching tables in addition to the directly matching ones. The challenge is to be robust to spuriously matched tables: we address it by developing a holistic matching framework based on topic sensitive pagerank and an augmentation framework that aggregates predictions from multiple matched tables. We propose a novel architecture that leverages preprocessing in MapReduce to achieve extremely fast response times at query time. Our experiments on real-life datasets and 573M web tables show that our approach has (i) significantly higher precision and coverage and (ii) four orders of magnitude faster response times compared with the state-of-the-art approach.
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
├─── img                                             # picture of model
├─── CreatIndex.py                                   # cerat KIV 、KIA 、Inverted index、docnum, etc                     
| 
├─── binaryGraphMatch.py                             # binaryGrapthMatch to achieve union base on join               
|─── changeParamiter_PPR_opendata.py                 # full ppr for opendata set 
|─── changeParamiter_PPR_webtable.py                 # full ppr for webtable set 
|─── creat_topk_join.py                              #  get topk for querying about join
|─── creat_topk_union.py                             #  get topk for querying about union
|   
├─── join.py             # join                                
|─── join_creatofflineIndex_webtable_opendata.py     # creat offline_index for join
|─── join_queryonline_webtable.py                    # query online for webtable
|─── join_queryonline_opendata.py                    # query online for opendata
|─── join_creat_topkfile.py                          # get topkfile for join
|─── join_staticdata_webtable_opendat.py             # stati cdata
|
├─── union.py      
├─── union.py            # union                                
|─── union_webtable.py                               # union on webtable
|─── union_opendata.py                               # union on opendata
|─── union_creat_topkfile.py                         # get topkfile about union
|─── union_staticdata_opendata.py                    # static data for opendata
|─── union_staticdata_webtable.py                    # static data for webtable
|
├─── util.py                                         # common functions
├─── page_ranks.py                                   # pageranks
├─── queryOnline.py                                  # query
├─── querydata.py                                    # process query
├─── staticdata.py                                   # static data
├─── staticdata_union_opendat.py                     
├─── staticdata_union_webtable.py                  
├─── myLogger.py                                     # log file
├─── info.md                                         # readme file

```

<br>



<span id="-getstart"></span>

## 🐳 Getting Started

This is an example of how to set up infogather locally. To get a local copy up, running follow these simple example steps.

### Prerequisites

infogather is bulit on pytorch, with torchvision, torchaudio, and transfrmers.

To insall the required packages, you can create a conda environmennt:

```sh
conda create --name info_env python=3.
```

then use pip to install -r requirements.txt

```sh
pip install -r requirements.txt
```


<span id="-quickstart"></span>

## 🐠 Quick Start

infogather is easy to use and extend. Going through the bellowing examples will help you familiar with infogather for quick use, evaluate an existing join/union algorithm on your own dataset, or developing new join/union algorithms.

**Step1: Check your environment**

You need to properly install nvidia driver first. To use GPU in a docker container You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)). Then, Please check your CUDA version via `nvidia-smi`

**Step2: index**

```sh
python join_creatofflineIndex_webtable_opendata.py 
-- datalist: list, dataset list
-- indexstorepath: string, the path of storing index  
-- columnValue_rate: float, the columnValue importance of the column  
-- columnName_rate :  float, the columnName importance of the column  
-- columnWith_rate : float, the columnWith importance of the column  
-- dataset_large_or_small: sting , large or small  
-- num_walks: int, the superparameter of ppr  
-- reset_pro: float,the superparameter of ppr  
```

**Step3: online**

```sh
# online:  
script: join_queryonline_opendata.py/join_queryonline_webtable.py    
run commond: python join_creatofflineIndex_webtable_opendata.py  

* parameters  
-- queryfilepath:string the querytablefilepath
-- columnname: the query column
```

**Step4: get_topk**
```sh
# get topk:  

topk: join_creat_topkfile.py/join_creat_topkfile.py  
script: python join_creat_topkfile.py  
run commond: python join_creatofflineIndex_webtable_opendata.py  
* parameters：  
-- filepath: string,the index of final_res_dic.pkl filepath  
-- storepath: string, the result of topk file store path  
```
<br>























