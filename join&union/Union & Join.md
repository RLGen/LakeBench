# Semantics-aware Dataset Discovery from Data Lakes with Contextualized Column-based Representation Learning

![The overall architecture of Starmie](starmie_overall.jpg)

### Requirements

TODO: list the requirements
* Python 3.7
* TODO add pytorch and CUDA version
* TODO add transformers version

Install requirements:
```
pip install -r requirements
```

### Datasets

TODO: list the datasets and download instructions

Datasets for table union search:
* Santos:
* TUS:

WDC web tables:
* See download instructions [here](https://webdatacommons.org/webtables/) for the 50M relational English tables.

Viznet (for column clustering and ML data discovery):
* TODO

### Running the offline pre-training pipeline:

The main entry point is `run_pretrain.py`. Example command:

```
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
  --task viznet \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 3 \
  --max_len 128 \
  --size 10000 \
  --projector 768 \
  --save_model \
  --augment_op drop_col \
  --fp16 \
  --sample_meth head \
  --table_order column \
  --run_id 0
```

Hyperparameters:

* `--task`: the tasks that we current support include "santos", "santosLarge", "tus", "tusLarge", "large", "small", "viznet". The tasks "large" and "small" are for column matching and "viznet" is for column clustering.
* `--batch_size`, `--lr`, `--n_epochs`, `--max_len`: standard batch size, learning rate, number of training epochs, max sequence length
* `--lm`: the language model (we use roberta for all the experiments)
* `--size`: the maximum number of tables/columns used during pre-training
* `--projector`: the dimension of projector (768 by default, same in all the experiments)
* `--save_model`: if this flag is on, the model checkpoint will be saved to the directory specified in the `--logdir` flag, such as `"results/viznet/model_drop_col_head_column_0.pt"`
* `--augment_op`: augmentation operator for contrastive learning. It includes `["drop_col", "sample_row", "sample_row_ordered", "shuffle_col", "drop_cell", "sample_cells", "replace_cells", "drop_head_cells", "drop_num_cells", "swap_cells", "drop_num_col", "drop_nan_col", "shuffle_row"]`
  1. Column-level: `drop_col` (drops a random column), `shuffle_col` (shuffles columns), `drop_num_col` (drops random numeric columns), `drop_nan_col` (drops columns with mostly NaNs)
  2. Row-level: `sample_row` (sample rows), `sample_row_ordered` (sample rows but preserve order), `shuffle_row` (shuffles the order of rows)
  3. Cell-level: `drop_cell` (drops a random cell), `sample_cells` (sample cells), `replace_cells` (sample random cells and replace with first ordered cells), `drop_head_cells` (drop first quarter cells), `drop_num_cells` (drop a sample of numeric cells), `swap_cells` (swap two cells)
* `--sample_meth`: table pre-processing operator that preserves order and de-duplicates. It includes `["head", "alphaHead", "random", "constant", "frequent", "tfidf_token", "tfidf_entity", "tfidf_row", "pmi"]`
  1. Row-level: `tfidf_row` (takes the rows with highest average tfidf scores), `pmi` (get highest pmi of pairs of column with topic column)
  2. Entity-level: `tfidf_entity` (takes entities in a column with highest after tfidf scores over its tokens)
  3. Token-level: `head` (take first N tokens), `alphaHead` (take first N sorted tokens), `random` (randomly sample tokens), `constant` (take every Nth token), `frequent` (take most frequently-occurring tokens), `tfidf_token` (take tokens with highest tfidf scores)
* `--fp16`: half-precision training (always turn this on)
* `--table_order`: row or column order for pre-processing, "row" or "column"
* `--single_column`: if this flag is on, then it will run the single-column variant ignoring all the
table context
* `--mlflow_tag`: use this flag to assign any additional tags for mlflow logging

### Model Inference:
Run `extractVectors.py`. Example command:

```
python extractVectors.py \
  --benchmark santos \
  --table_order column \
  --run_id 0
```

Hyperparameters
* `--benchmark`: the current benchmark for the experiment. Examples include `santos`, `santosLarge`, `tus`, `tusLarge`, `wdc`
* `--single_column`: if this flag is on, then it will retrieve the single-column variant
* `--run_id`: the run_id of the job (I use 0 for experiments)
* `--table_order`: column-ordered or row-ordered (always use `column`)
* `--save_model`: whether to save the vectors in a pickle file, which is then used in the online processing


### Online processing

1. Linear & Bounds: Run `test_naive_search.py`. Some scripts are in `tus_cmd.sh` and `run_tus_all.py` (for slurm scheduling). Example  command:

```
python test_naive_search.py \
  --encoder cl \
  --benchmark santos \
  --augment_op drop_col \
  --sample_meth tfidf_entity \
  --matching linear \
  --table_order column \
  --run_id 0 \
  --K 10 \
  --threshold 0.7
```

Hyperparameters
* `--encoder`: choice of encoder. Options include "cl" (this is for both full Starmie and
singleCol baseline), "sato", "sherlock"
* `--benchmark`: choice of benchmark for data lake. Options include "santos", "santosLarge",
"tus", "tusLarge", "wdc"
* `--augment_op`: choice of augmentation operator
* `--sample_meth`: choice of sampling method
* `--matching`: "linear" matching (full) or "bounds". If you would like to run "greedy", add the
function call to the code
* `--table_order`: "column" or "row" (just use column)
* `--run_id`: always 0
* `--single_column`: when set to True, run the single column baseline
* `--K`: what you would like to set K to in top-K results
* `--threshold`: the similarity threshold

FOR ERROR ANALYSIS: bucket (bucket number between 0 and 5), analysis (either "col" for number of columns, "row" for number of rows,numeric" for percentage of numerical columns

FOR SCALABILITY EXPERIMENTS: scal (what fraction of data lake do we want to get the metrics scores for – 0.2,0.4,0.6,0.8,1.0)



2. LSH: Run test_lsh.py (example script: lsh_cmd.sh). Example command:

```
python test_lsh.py \
--encoder cl \
--benchmark santosLarge \
--run_id 0 \
--num_func 8 \
--num_table 100 \
--K 60 \
--scal 1.0
```

Hyperparameters:
* `--encoder`: choice of encoder. Options include "cl" (this is for both full Starmie and
singleCol baseline), "sato", "sherlock"
* `--benchmark`: choice of benchmark for data lake. Options include "santos", "santosLarge",
"tus", "tusLarge", "wdc"
* `--run_id`: always 0
* `--single_column`: when set to True, run the single column baseline
* `--num_func`: number of hash functions (always use 8 for ‘cl’ encoder)
* `--num_table`: number of tables (always use 100 for ‘cl’ encoder)
* `--K`: what you would like to set K to in top-K results

FOR SCALABILITY EXPERIMENTS: scal (what fraction of data lake do we want to get
the metrics scores for – 0.2,0.4,0.6,0.8,1.0)

3. HNSW: Run test_hnsw_search.py (example script: hnsw_cmd.sh).
Example command:
```
python test_hnsw_search.py \
--encoder cl \
--benchmark santosLarge \
--run_id 0 \
--K 60 \
--scal 1.0
```

Hyperparameters:

* `--encoder`: choice of encoder. Options include "cl" (this is for both full Starmie and
singleCol baseline), "sato", "sherlock"
* `--benchmark`: choice of benchmark for data lake. Options include "santos", "santosLarge",
"tus", "tusLarge", "wdc"
* `--run_id`: always 0
* `--single_column`: when set to True, run the single column baseline
* `--K`: what you would like to set K to in top-K results

FOR SCALABILITY EXPERIMENTS: scal (what fraction of data lake do we want to get
the metrics scores for – 0.2,0.4,0.6,0.8,1.0)



## Data discovery for ML tasks:

Run `discovery.py`. We assume:
1. A model checkpoint in `results/viznet/model_drop_col_head_column_0.pt`
2. The viznet dataset in `data/viznet/`

Run the script by
```
python discovery.py
```
The code will print out the MSE for NoJoin, contrastiving learning (CL), Jaccard, and Overlap. The joined tables will be output to pickled files named `none_joined_tables.pkl`, `cl_joined_tables.pkl`, `jaccard_joined_tables.pkl`, and `overlap_joined_tables.pkl`.

### Column clustering:

See Line 273 and Line 128 of the file `sdd/pretrain.py`.
To run column clustering, you can run a sequence of commands (remember to check the file paths):

```
CUDA_VISIBLE_DEVICES=7 python run_pretrain.py \
  --task viznet \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 3 \
  --max_len 128 \
  --size 10000 \
  --projector 768 \
  --save_model \
  --augment_op drop_col \
  --fp16 \
  --sample_meth head \
  --table_order column \
  --run_id 0
```

Copy the clustering results:
```
cp *.pkl data/viznet/multi_column
```

Each run will pre-train the models on 10k viznet tables and cluster all the columns. The clustering results will be stored at `data/viznet/multi_column/clusters.pkl` and `data/viznet/single_column/`.

To view the clusters, you can use the jupyter notebook in `notebook/offline.ipynb`. Running the last cell should print out some clusters like

```
artist ---- 1. I Don&#39;t Give A ...; 2. I&#39;m The Kinda; 3. I U She; 4. Kick It [featuring Iggy Pop]; 5.
Operate
artist ---- 1. Spoken Intro; 2. The Court; 3. Maze; 4. Girl Talk; 5. A La Mode
artist ---- 1. Street Fighting Man; 2. Gimme Shelter; 3. (I Can&#39;t Get No) Satisfaction; 4. The
Last Time; 5. Jumpin&#39; Jack Flash
…
---------------------------------
type ---- Emerson Elementary School; Banneker Elementary School; Silver City Elementary
School; New Stanley Elementary School; Frances Willard Elementary School
type ---- Choctawhatchee Senior High School; Fort Walton Beach High School; Ami Kids
Emerald Coast; Gulf Coast Christian School; Adolescent Substance Abuse
city ---- Chilton; Stoughton
…
---------------------------------
description ---- Fri Sep 11,2015 3:30 PM (CST); Fri Sep 11,2015 6:00 PM (CST); Sat Sep
12,2015 10:00 AM (CST); Sat Sep 12,2015 12:00 PM (CST); Sat Sep 12,2015 5:30 PM (CST)
day ---- Sept. 1; Sept. 7; Sept. 22; Sept. 29; Oct. 5
description ---- Fri Sep 11,2015 3:30 PM (CST); Fri Sep 11,2015 6:00 PM (CST); Sat Sep
12,2015 10:00 AM (CST); Sat Sep 12,2015 12:00 PM (CST); Sat Sep 12,2015 5:30 PM (CST)
...
address ---- 1721 Papillon St, North Port FL; 4113 Wabasso Ave, North Port FL; 3681
Wayward Ave, North Port FL; 1118 N Salford Blvd, North Port FL; 2057 Bendix Ter, North
Port FL
address ---- 5 Brand Rd, Toms River NJ; 40 12th St, Toms River NJ; 75 Sea Breeze Rd,
Toms River NJ; 98 Oak Tree Ln, Toms River NJ; 67 16th St, Toms River NJ
address ---- 652 Martha St, Montgomery AL; 3184 Lexington Rd, Montgomery AL; 120 S
Lewis St, Montgomery AL; 1812 W 2nd St #OP, Montgomery AL; 3582 Southview Ave,
Montgomery AL
---------------------------------
```
