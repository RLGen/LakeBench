<div>
    <h1>Pexeso</h1>
</div>
![image-20231121160556291](C:\Users\w2306\AppData\Roaming\Typora\typora-user-images\image-20231121160556291.png)

<h2>Quik Start</h2>

**Step1: Check your environment**

You need to properly install python package first. Please check package you have installed via `pip list`

**Step2: Runing**

```sh
#webtable
python PEXESO_benchmark_ws.py
#opendata
python PEXESO_benchmark_os.py
```

**Step3: Output**

Automatically check if the index has been generated. If it is complete, start querying and output the result file to the path of `storage_path`.

### functional module

##### Pexeso\block_and_verify.py： 

Block and verify and citation implementation in thesis

##### Pexeso\Hierarchical_Grid.py：

Implementation code for the hierarchical grid class in the index

##### Pexeso\Pivot_Metric.py：

Selection of pivot algorithm code

##### Pexeso\inverted_index.py：

Implementation code for the inverted index class in Index

### Parameter settings

k：Number of pivots

n_layers：The number of layers of the hierarchical grid index.

tao：Distance threshold

T：Column joinability threshold

n_dims：Dimension of the embedding

### Output files and intermediate files

dic_xx.pickle：Intermediate files, which preprocess the input data into embedding form, select pivots and map them, store the preprocessed data; and save the generated indexes. 

Pe_results_xx: Output folder containing csv files of topk results.
