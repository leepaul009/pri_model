# pri_model


## 1. Setup conda environment
Inside repo, find deps.yml file that includes all the dependent python packages or other softwares.  
Create conda environment with deps.yml file by using following command:
```
# {YOUR USER DIR} and {NAME OFYOUR ENV} depends on your PC.
conda env create -f deps.yml -p /home/{YOUR USER DIR}/anaconda3/envs/{NAME OFYOUR ENV}
```

## 2. Dataset
### 2.1 Directory of dataset
Dataset files of sequence should be csv file.  
Dataset files of sequence is put in the following directory:
```
data/train/
data/val/
data/test/

# For example we could have several csv files in training directory, file1.csv, file2.csv, ... (all files are sequence data)
# Then all the csv files will be use to produce training data.
```
Sequence data should be stored in csv file.  
Notice! following column must exist in csv file:
- Amino acid sequence in the 'protein_sequence' column.  
- DNA/RNA sequence in the 'nucleotide_sequence' column.  
- Experimental affinity value in the 'dG' column.  

### 2.2 Description of dataset processing function
Class PriDataset will process all the dataset files in __init__ method.
Function pri_get_instance is the function to process one complex(protein-DNA/RNA) to desired input for training.

## 3. Train and inference
### 3.1 Train from beginning
```
python train.py --data_dir data/train --data_dir_for_val data/val --core_num 8 --output_dir output --train_batch_size 40 --num_train_epochs 100 --do_eval

# parameter:
#   data_dir: directory of training dataset
#   data_dir_for_val: directory of validation dataset
#   core_num: how many cores used to process data
#   output_dir: directory where model will be stored
#   train_batch_size: how may complex trained in one step
#   num_train_epochs: number of training epochs
#   do_eval
```

### 3.2 Train from pretrained model, also named five-tuning
```
python train.py --data_dir data/train --data_dir_for_val data/val --core_num 8 --output_dir output --train_batch_size 40 --num_train_epochs 100 --do_eval --resume --resume_path output/model.99.202.ckpt 

# parameter:
#   resume: set true to use resume model parameter
#   resume_path: directory of resume model parameter
```



