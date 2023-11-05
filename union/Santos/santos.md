# Santos

## Repository Organization

\- **benchmark**  folder contains subfolders for SANTOS Small Benchmark (santos_benchmark), SANTOS Large Benchmark (real_data_lake_benchmark) and TUS Benchmark (tus_benchmark).

\- **codes** folder contains SANTOS source codes for preprocessing yago, creating synthesized knowledge base, preprocessing data lake tables using yago and querying top-k SANTOS unionable tables.

\- **groundtruth** folder contains the groundtruth files used to evaluate precision and recall.

\- **hashmap** folder contains indexes built during the preprocessing phase.

\- **images** folder contains supplementary images submitted with the paper.

\- **stats** folder contains SANTOS output files related to top-k search results and efficiency.

\- **yago** folder contains the original and indexed yago files.

\- **README.md** file explains the repository.

\- **requirements.txt** file contains necessary packages to run the project.


## Reproducibility

1. Download, unzip and upload [YAGO](https://yago-knowledge.org/downloads/yago-4) knowledge base to [yago/yago_original](yago/yago_original) folder.
2. Run [preprocess_yago.py](codes/preprocess_yago.py) to create entity dictionary, type dictionary, inheritance dictionary and relationship dictionary. Then run [Yago_type_counter.py](codes/Yago_type_counter.py), [Yago_subclass_extractor.py](codes/Yago_subclass_extractor.py) and [Yago_subclass_score.py](codes/Yago_subclass_score.py) one after another to generate the type penalization scores. The created dictionaries are stored in [yago/yago_pickle](yago/yago_pickle/). You may delete the downloaded yago files after this step as we do not need orignal yago in [yago/yago_original](yago/yago_original) anymore.
3. Run [data_lake_processing_yago.py](codes/data_lake_processing_yago.py) to create yago inverted index.
4. Run [data_lake_processing_synthesized_kb.py](codes/data_lake_processing_synthesized_kb.py) to create synthesized type dictionary, relationship dictionary and synthesized inverted index.
5. Run [query_santos.py](codes/query_santos.py) to get top-k SANTOS unionable table search results.

