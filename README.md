# UA-RAG: Uncertainty-Aware Dynamic Retrieval-Augmented Generation.

## Install environment

```bash
conda create -n ua-rag python=3.9
conda activate ua-rag
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run UA-RAG

### Build index

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

### Download Dataset

For 2WikiMultihopQA:

Download the [2WikiMultihop](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository <https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1>. Unzip it and move the folder to `data/2wikimultihopqa`.

For StrategyQA:

```bash
wget -O data/strategyqa_dataset.zip https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
mkdir -p data/strategyqa
unzip data/strategyqa_dataset.zip -d data/strategyqa
rm data/strategyqa_dataset.zip 
```

For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

For IIRC:

```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```

For SQuAD:

```bash
mkdir -p data/squad
wget -O data/squad/dev-v1.1.json https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/refs/heads/master/dataset/dev-v1.1.json
```

### Run

```bash
python main.py -c config/config_name
```

### Evaluate

```bash
python evaluate.py --dir path_to_folder(result/2wikimultihopqa_llama2_13b/fold)
```
## Acknowledgment

This project builds upon the foundation of [DRAGIN](https://github.com/oneal2000/DRAGIN), with enhancements and adaptations implemented in accordance with the Apache License 2.0.
