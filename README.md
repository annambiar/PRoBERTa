
# PRoBERTa
Ananthan Nambiar, Maeve heflin, Simon Liu, Mark Hopkins, Sergei Maslov, Anna Ritz

## Notes
- Links to Google Drive folders are provided for example output and data files.

## Requirements and Installation
sentencepeice tokenizer
```bash
pip3 install sentencepeice
```
Build [fairseq from linked repo source](https://github.com/imonlius/fairseq.git).
```bash
git clone https://github.com/imonlius/fairseq.git
cd fairseq
pip3 install --editable . --no-binary cffi
```
### tokenizer.py
Train a tokenizer and tokenize data for protein family and interaction fine-tuning

#### Example usage:
```bash
python3 tokenizer.py
```
- To change (if needed)
	* path: path to the protein family data. This should be a .tab file with "Sequence" and "Protein families" as two of the columns
	* int_path: path to protein interaction data. This should be a json file with 'from', 'to' and 'link' for each interaction

### pRoBERTa_pretrain.sh
 Pre-train RoBERTa model

#### Example Usage:
```bash
bash pRoBERTa_pretrain.sh pretrain 4 /bert/pretrained_model \
        /bert/data/pretraining/split_binarized/ \
        768 5 125000 3125 0.0025 32 64 3
```
- Arguments
	* PREFIX: pretrain
	* NUM_GPUS: 4
	* OUTPUT_DIR: [/bert/pretrained_model](https://drive.google.com/drive/u/2/folders/1fyb3RklnVWAUwajv20BP5smq9ypDgMl9)
	* DATA_DIR: [/bert/data/pretraining/split_binarized/](https://drive.google.com/drive/u/2/folders/1inKxRuf5f3JBM2YDO1dQc-gTsdMn6VGR)
	* ENCODER_EMBED_DIM: 768
	* ENCODER_LAYERS: 5
	* TOTAL_UPDATES: 125000
	* WARMUP_UPDATES: 3125
	* PEAK_LEARNING_RATE: 0.0025
	* MAX_SENTENCES: 32
	* UPDATE_FREQ: 64
	* PATIENCE: 3

### protein_family_clustering.py
Cluster proteins using k-means and calculate the normalized mutual information (NMI) with pritein families

#### Example Usage:
```bash
python3 protein_family_clustering.py
```
- To change (if needed)
	* tokenized_data_filepath: input data filepath. This file has to contain tokenized protein sequences in a 'Tokenized Sequence' column, and the family each protein belongs to in a 'Protein families' column. Any other columns in this file will be ignored.
	* roberta_weights: depending on whether you're using a pretrained or fine-tuned model, choose the appropriate weights
	* EMBEDDING_SIZE: To match the PRoBERTa model size
	* USE_NULL_MODEL: to use random cluster prediction instead of k-means clustering.


### pRoBERTa_finetune_ppi.sh: 
Fine-tune RoBERTa model for Protein Interaction Prediction Task

#### Example Usage:
```bash
bash pRoBERTa_finetune_ppi.sh ppi 4 /bert/ppi_prediction \
        /bert/data/ppi_prediction/split_binarized/robustness_minisplits/0.80/ \
        768 5 12500 312 0.0025 32 64 2 3 \
        /bert/remastered_run/pretraining/checkpoint_best.pt \
        no
```
- Arguments
	* PREFIX: ppi
	* NUM_GPUS: 4
	* OUTPUT_DIR: [/bert/ppi_prediction](https://drive.google.com/drive/u/2/folders/1mS34_2YTBh2wZuvn9QF7m0254bnc2LE_)
	* DATA_DIR: [/bert/data/ppi_prediction/split_binarized/robustness_minisplits/1.00](https://drive.google.com/drive/u/2/folders/1kjNnud51AIPu_eeuqdapHHE-GVoaHfZm)
	* ENCODER_EMBED_DIM: 768
	* ENCODER_LAYERS: 5
	* TOTAL_UPDATES: 12500
	* WARMUP_UPDATES: 312
	* PEAK_LEARNING_RATE: 0.0025
	* MAX_SENTENCES: 32
	* UPDATE_FREQ: 64
	* NUM_CLASSES: 4083
	* PATIENCE: 3
	* PRETRAIN_CHECKPOINT: [/bert/remastered_run/pretraining/checkpoint_best.pt](https://drive.google.com/drive/u/2/folders/1TbFjyRfbkLgJ_rlvO1SFB-ZvwQyykvK7)
	* RESUME_TRAINING: no

### pRoBERTa_finetune_pfamclass.sh:
Fine-tune RoBERTa model for Family Classification Task

#### Example Usage:
```bash
bash pRoBERTa_finetune_pfamclass.sh family 4 /bert/family_classification \
        /bert/data/family_classification/split_binarized/robustness_minisplits/1.00 \
        768 5 12500 312 0.0025 32 64 4083 3 \
        /bert/remastered_run/pretraining/checkpoint_best.pt \
        no
```
- Arguments
		* PREFIX: family
		* NUM_GPUS: 4
		* OUTPUT_DIR: [/bert/family_classification](https://drive.google.com/drive/u/2/folders/1EGvJEAVDfPb1gcxPUsr92Tan9rPgasGm)
		* DATA_DIR: [/bert/data/family_classification/split_binarized/robustness_minisplits/1.00](https://drive.google.com/drive/u/2/folders/1VxNHbwWqVZsnnZwA-6gjFtxkXB55tX3y)
		* ENCODER_EMBED_DIM: 768
		* ENCODER_LAYERS: 5
		* TOTAL_UPDATES: 12500
		* WARMUP_UPDATES: 312
		* PEAK_LEARNING_RATE: 0.0025
		* MAX_SENTENCES: 32
		* UPDATE_FREQ: 64
		* NUM_CLASSES: 4083
		* PATIENCE: 3
		* PRETRAIN_CHECKPOINT: [/bert/remastered_run/pretraining/checkpoint_best.pt](https://drive.google.com/drive/u/2/folders/1TbFjyRfbkLgJ_rlvO1SFB-ZvwQyykvK7)
		* RESUME_TRAINING: no

#### pRoBERTa_evaluate_family_batch.py: 
Predict families using fine-tuned RoBERTa model

#### Example Usage:
```bash
python3 pRoBERTa_evaluate_family_batch.py /bert/data/family_classification/split_tokenized/full/Finetune_fam_data.split.test.10 \
	/bert/data/family_classification/split_binarized/robustness_minisplits/1.00/ \
	predictions.tsv \
	/bert/family_classification/checkpoints/ \
	protein_family_classification 256
```
- Arguments
		* DATA: [/bert/data/family_classification/split_tokenized/full/Finetune_fam_data.split.test.10](https://drive.google.com/drive/u/2/folders/1CvZPrtqs_JqxJVG3Fk-2FwUEC5R7NNUU)
		* BINARIZED_DATA: [/bert/data/family_classification/split_binarized/robustness_minisplits/1.00/](https://drive.google.com/drive/u/2/folders/1VxNHbwWqVZsnnZwA-6gjFtxkXB55tX3y)
		* OUTPUT: [predictions.tsv](https://drive.google.com/drive/u/2/folders/10gpJUzyjPCT12GfqUexOFcjFoTW9Rcr4)
		* MODEL_FOLDER: [/bert/family_classification/checkpoints/](https://drive.google.com/drive/u/2/folders/1JgEfybT6wT8MGzaxgAUKI7dH6W0oWLBn)
		* CLASSIFICATION_HEAD_NAME: protein_family_classification
		* BATCH_SIZE: 256

### pRoBERTa_evaluate_ppi_batch.py: 
Predict PPI using fine-tuned RoBERTa model

#### Example Usage:
```bash
python3 pRoBERTa_evaluate_ppi_batch.py /bert/data/ppi_prediction/split_tokenized/full/Finetune_interact_tokenized.split.test.10 \
	/bert/data/ppi_prediction/split_binarized/robustness_minisplits/1.00/ \
	predictions.tsv \
	/bert/ppi_prediction/checkpoints/ \
	protein_interaction_prediction 256
```
- Arguments:
		* DATA: [/bert/data/ppi_prediction/split_tokenized/full/Finetune_interact_tokenized.split.test.10](https://drive.google.com/drive/u/2/folders/1GxGGOqQz5LvlLoTW3EnuEEr7fKwmu8ju)
		* BINARIZED_DATA: [/bert/data/ppi_prediction/split_binarized/robustness_minisplits/1.00/](https://drive.google.com/drive/u/2/folders/1kjNnud51AIPu_eeuqdapHHE-GVoaHfZm)
		* OUTPUT: [predictions.tsv](https://drive.google.com/drive/u/2/folders/1mS34_2YTBh2wZuvn9QF7m0254bnc2LE_)
		* MODEL_FOLDER: [/bert/ppi_prediction/checkpoints/](https://drive.google.com/drive/u/2/folders/1PvcqbJbgjUNMgoYhTNCsZ_a2oEAIBjxV)
		* CLASSIFICATION_HEAD_NAME: protein_interaction_prediction
		* BATCH_SIZE: 256
