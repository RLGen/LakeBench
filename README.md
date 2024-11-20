<div align= "center">
    <h1> LakeBench: A Benchmark for Discovering Joinable and Unionable Tables in Data Lakes</h1>
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
<img src="imgs/framework.png" width="1000px">
</div>
<br>

🌊  LakeBench is a large-scale benchmark designed to test the mettle of *table discovery* methods on a much larger scale, providing a more comprehensive and realistic evaluation platform for the field, including *finance, retail, manufacturing, energy, media, and more.*

  Despite their paramount significance, existing benchmarks for evaluating and supporting *table discovery* processes have been limited in scale and diversity, often constrained by small dataset sizes. They are not sufficient to systematically evaluate the effectiveness, scalability, and efficiency of various solutions.

  LakeBench consists of over 16 million real tables **–1,600X** larger than existing data lakes, from multiple sources, with an overall size larger than 1TB (**100X** larger). LakeBench contains both synthesized and real queries, in total more than 10 thousand queries –**10X** more than existing benchmarks, for join and union search respectively. 

🙌  With LakeBench, we thoroughly evaluate the state-of-the-art *table discovery* approaches on our benchmark and present our experimental findings from diverse perspectives, which we believe can push the research of *table discovery*.

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

LakeBench is built on pytorch, with torchvision, torchaudio, and transformers.

To install the required packages, you can create a conda environment:

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
|        [WebTable](https://drive.google.com/file/d/1tnI2EyrYHlc3fpv0SSMoe2sqWQZoOEjg/view?usp=drive_link)     |  [WebTable_Union_Query](https://drive.google.com/drive/folders/1mvEpyia9e8S365Ld8mYQbSactagmry43)     |  [WebTable_Union_Groun_Truth](https://drive.google.com/drive/folders/1p8ke4R32TkkZP5aHXxhxj_J6qr1FnVIq)     | 
|       [OpenData_SG](https://drive.google.com/file/d/1pPKMJ2Xnd6gYtkT_zVHIHCC97K5Yib4e/view?usp=drive_link)       |   [WebTable_Join_Query](https://drive.google.com/drive/folders/1MRQd1iTJTvNHZtQABzW2ZQwh9rF49HFD)       |    [WebTable_Join_Groun_Truth](https://drive.google.com/drive/folders/1p8ke4R32TkkZP5aHXxhxj_J6qr1FnVIq)       |  
|       [OpenData_CAN](https://drive.google.com/file/d/1ksOyaGVugeu7UJ0SKbYj4ri-rwgGfNH8/view?usp=drive_link)       |    [OpenData_Union_Query](https://drive.google.com/drive/folders/1DV3y3Drv3BWP8noRW-nP_i9RPV_9-q8P)       |  [OpenData_Union_Groun_Truth](https://drive.google.com/drive/folders/1p8ke4R32TkkZP5aHXxhxj_J6qr1FnVIq)       |  
|       [OpenData_UK、OpenData_USA](https://drive.google.com/drive/folders/1F9hIN815B6jmn85t-4gQGDoV-gLX8QvH?usp=drive_link)       |    [OpenData_Join_Query](https://drive.google.com/drive/folders/18EuWenSKSSaRKACo_tVcSV5_PrIvW3ql)       |   [OpenData_Join_Groun_Truth](https://drive.google.com/drive/folders/1p8ke4R32TkkZP5aHXxhxj_J6qr1FnVIq)       |   

<span id="-quickstart"></span>

## 🐠 Instruction

LakeBench is easy to use and extend. Going through the bellowing examples will help you familiar with LakeBench for detailed instructions, evaluate an existing join/union algorithm on your own dataset, or developing new join/union algorithms.

### Example
Here is an example to run InfoGather. Running other supported algorithms (on other datasets with different queries) can be specified by the <a href = "#-table">table</a> below.

<div align= "center">
    <h1> InfoGather-Entity Augmentation and Attribute Discovery By Holistic Matching with Web Tables</h1>
</div>
<p align="center">
  <a href="#-struct">Folder Structure</a> •
  <a href="#-getstart">GettingStart</a> •
  <a href="#-quickstart">QuickStart</a> •
  <a href="#-result">Result</a> •
</p>




<br>

<div align="center">
<img src="imgs/infogather.jpg" width="1000px">
<img src="imgs/info2.jpg" width="1000px">
</div>
<br>

🌊  The Web contains a vast corpus of HTML tables, specifically entityattribute tables. We present three core operations, namely entity augmentation by attribute name, entity augmentation by example and attribute discovery, that are useful for "information gathering" tasks (e.g., researching for products or stocks). We propose to use web table corpus to perform them automatically. We require the operations to have high precision and coverage, have fast (ideally interactive) response times and be applicable to any arbitrary domain of entities. The naive approach that attempts to directly match the user input with the web tables suffers from poor precision and coverage. Our key insight is that we can achieve much higher precision and coverage by considering indirectly matching tables in addition to the directly matching ones. The challenge is to be robust to spuriously matched tables: we address it by developing a holistic matching framework based on topic sensitive pagerank and an augmentation framework that aggregates predictions from multiple matched tables. We propose a novel architecture that leverages preprocessing in MapReduce to achieve extremely fast response times at query time. Our experiments on real-life datasets and 573M web tables show that our approach has (i) significantly higher precision and coverage and (ii) four orders of magnitude faster response times compared with the state-of-the-art approach.
<span id="-community"></span>

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

## 🐳 Instruction
Infogather is easy to use and extend. Going through the bellowing examples will help you familiar with infogather for detailed instructions, evaluate an existing join/union algorithm on your own dataset, or developing new join/union algorithms.

### Pre-requisites

Infogather is bulit on pytorch, with torchvision, torchaudio, and transfrmers.

To install the required packages, you can create a conda environment:

```sh
conda create --name info_env python=3.
```

then use pip to install the required packages

```sh
pip install -r requirements.txt
```


<span id="-quickstart"></span>

## 🐠 join


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



## 🐠 union

**Step1: Check your environment**

You need to properly install nvidia driver first. To use GPU in a docker container You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)). Then, Please check your CUDA version via `nvidia-smi`. Because we often get the results of union search based on the Bipartite Graph Matching on the results of join search, which is stored in `storepath (the join result of topk file store path)` 

**Step2: online**

```sh
# online:  
script: union_opendata.py/union_webtable.py
python union_opendata.py/union_webtable.py


```

**Step3: get_topk**
```sh
# get topk:  

topk: union_creat_topkfile.py
script: python union_creat_topkfile.py
run command: python python union_creat_topkfile.py
```

<br>



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


## 🐠 LakeCompass
We propose to build an end-to-end prototype system, LakeCompass that encapsulates the above functionalities through flexible Python APIs, where the user can upload data, construct indexes, search tables and build ML models. We also provide a Web interface to serve table search and analysis in a more intuitive way.

### Example
We support various types of indexes built over data lakes, enabling efficient table searches across three categories: keyword-based, joinable, and unionable.
First, we invoke ‘indexing’ function to build an index over data lake metadata. The arguments for this function include index type, search type, and an instance of a configuration class. Once the index is created, we utilize ‘keyword_search’ function to retrieve tables associated with the keyword ‘education’.
```sh
# build index for the datalake
datalake = LakeCompass,DataLake('/datalake demo')
keyword_index = datalake.metadata.indexing(index_type='HNSW', search_type='keyword', config=LakeCompass.HNSWconfig())

# keyword-based search
candidate_table = LakeCompass.keyword_search(keyword_index, 'education')
```

After selecting a table as the query table and training a model based solely on it, the results indicate that the performance of the model tends to be poor.
```sh
# train configuration
predict_col ='Educational Category'
model_config = {'model':'SvM','type': 'classification', 'k':4, 'model config': SVMconfig()}
query_table = LakeCompass.read('education_Des_Moines.csv')
query_table_model = DownstreamModel(**model_config)
val_set = pd.read csv('val set.csv')
test_set = pd.read csv('test set.csv')
# trian a model using query table
query_table_model.train(query_table, predict_col, val_set)
query_table_model.eval(test_set, predict_col)
```

Therefore, we proceed to employ the ‘union_search’ function to retrieve unionable tables based on semantics similarities. Before the search, we build another index over the columns within the data lake.
```sh
#unionable table search
union_index = datalake.columns.indexing(index_type='HNSW', search_type='union', config=LakeCompass.HNSWconfig())
unionable_tables = LakeCompass.union_search(union_index, query_table)
```

We further provides ‘auto_augment’ function that augments the original query table with the retrieved tables based on their benefits to the downstream model performance. Simultaneously,  the model is trained using the augmented table datasets. This augmentation and training process follows a novel iterative strategy. The results indicate that the model trained on the augmented table datasets outperforms the former model that trained solely on the query table.
```sh
#train another model using unionable tables
augmented_table_model = DownstreamModel(**model_config)
augmented_table_model = unionable_tables.auto_augment(augmented_table_model, val_set, predict_col)
# evaluate models
augmented_table_model.eval(test_set, predict_col)
```
