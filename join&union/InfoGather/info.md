






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
<img src="img/deepjoin.jpg" width="1000px">
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

This is an example of how to set up deepjoin locally. To get a local copy up, running follow these simple example steps.

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

Deepjoin is easy to use and extend. Going through the bellowing examples will help you familiar with Deepjoin for quick use, evaluate an existing join/union algorithm on your own dataset, or developing new join/union algorithms.

**Step1: Check your environment**

You need to properly install nvidia driver first. To use GPU in a docker container You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)). Then, Please check your CUDA version via `nvidia-smi`

**Step2: Pretrain**

```sh
python deepjoin_train.py --dataset opendata --opendata all-mpnet-base-v2 --model_save_path /deepjoin/model/output  
-- dataset [choose task, str] [opendata, webtable]
--opendata [train_model name str] [all-mpnet-base-v2]
--model_save_path [trained_model_save_path,str]
--file_train_path [pretain_file_path,str]
--train_csv_file [pretrain_file path str]
--storepath [pretrain index store path str]
```

**Step3: infer**

```sh
python deepjoin_infer.py 
-- dataset [choose task, str] [opendata, webtable]
--datafile [infer_tables_file ,str]
--storepath [final_reslut_storepath,str]
```

**Step4: Indexing**

Here are some parameters:

> --benchmark [Choose benchmark, str] [opendata, opendata_large, webtable, webtable_large]

```sh
python index.py --benchmark webtable
```

**Step5: Querying**

> --benchmark [Choose benchmark, str] [opendata, opendata_large, webtable, webtable_large]
>
> --K [Choose top-*k* ,int] [5~60]
>
> --threshold [Choose threshold, float] [0.5~1.0]
>
> --N [Choose N, int] [4, 10, 16, 25, 50]

```sh
python query.py --benchmark webtable --K 5 --N 10 --threshold 0.7
```

<br>















# infogather
Paper：InfoGather- Entity Augmentation and Attribute Discovery By Holistic Matching with Web Tables   code for join and union



# offline:
python join_creatofflineIndex_webtable_opendata.py 

script: join_creatofflineIndex_webtable_opendata.py  
run commond: python join_creatofflineIndex_webtable_opendata.py
* parameters:  
    * 1 datalist: list, dataset list   
    * 2 indexstorepath: string, the path of storing index  
    * 3 columnValue_rate: float, the columnValue importance of the column  
    * 4  columnName_rate :  float, the columnName importance of the column  
    * 5 columnWith_rate : float, the columnWith importance of the column  
    * 6 dataset_large_or_small: sting , large or small  
    * 7 num_walks: int, the superparameter of ppr  
    * 8 reset_pro: float,the superparameter of ppr  


# online:  

location：13022  /home/lijiajun/infogather    
script: join_queryonline_opendata.py/join_queryonline_webtable.py    

run commond: python join_creatofflineIndex_webtable_opendata.py  

* parameters  
  * 1 queryfilepath:string the querytablefilepath
  * 2 columnname: the query column  

# get topk:  

topk: join_creat_topkfile.py/join_creat_topkfile.py  
location：13022  /home/lijiajun/infogather  
script: python join_creat_topkfile.py  
run commond: python join_creatofflineIndex_webtable_opendata.py  
* parameters：  
  * 1 filepath: string,the index of final_res_dic.pkl filepath  
  * 2 storepath: string, the result of topk file store path  









