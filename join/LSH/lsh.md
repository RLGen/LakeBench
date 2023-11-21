<div>
    <h1>LSH Ensemble</h1>
</div>


<h2>Quik Start</h2>

**Step1: Check your environment**

You need to properly install python package first. Please check package you have installed via `pip list`

**Step2: Runing**

```sh
#webtable
python LSH_benchmark_ws.py
#opendata
python LSH_benchmark_os.py
```

**Step3: Output**

Automatically check if the index has been generated. If it is complete, start querying and output the result file to the path of `storage_path`.


### functional module

##### LSH Ensemble\datasketch\lsh.py:  

Implementation code for MinHashLSH class

##### LSH Ensemble\datasketch\minhash.py: 

Implementation code for Minhash class

##### LSH Ensemble\datasketch\storage.py: 

LSHensemble's underlying storage related

##### LSH Ensemble\datasketch\lshensemble.py: 

Implementation code for MinHashLSHEnsemble class

**Main functions**

Initialization: Load parameters. Input - Containment threshold, num_ Perm, num_ Part, m, weights

index: Indexing, only called once. Input - The candidate set represented by the list of (key, minhash, size)

query: Conduct a query. Input: minhash and size of the query set

##### LSH Ensemble\datasketch\lshensemble_partition.py

Functional module for calculating optimal partition

##### LSH Ensemble\lshensemble_benchmark.py

This section is the program entry, and when executed with the parameter level test/lite, the corresponding parameter set can be used. The parameter set is set in the code.


### Parameter settings

threshold: Containment threshold Value range[0.0, 1.0].

num_perm: The number of permutation functions used in Minhash.

num_part: the number of partitions of LSH Ensemble.

m: LSH Ensemble uses approximately m times more memory space than the same number of MinHash LSHs. In addition, the larger the m, the higher the accuracy.

weights: When optimizing parameter settings, balance the importance of fp and fn.

### Output files and intermediate files

index.pickle, query.pickle: Intermediate file, which preprocesses the input index and query data into (key, minhash, size) format for storage.

Lsh_results_xx: output folder containing csv files of topk results.