# pri_model


## 1. Setup conda environment
Inside repo, find deps.yml file that includes all the dependent python packages or other softwares.  
Create conda environment with deps.yml file by using following command:
```
# {YOUR USER DIR} and {NAME OFYOUR ENV} depends on your PC.
conda env create -f deps.yml -p /home/{YOUR USER DIR}/anaconda3/envs/{NAME OFYOUR ENV}



pip install axial-positional-embedding
```

## 2. Dataset
### 2.1 Directory of dataset
Dataset files of sequence should be csv file.  
Dataset files of sequence is put in the following directory:
```
dataset/pri_data/train/
dataset/pri_data/val/
dataset/pri_data/test/

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
python train.py --data_dir dataset/pri_data/train --data_dir_for_val dataset/pri_data/val --core_num 8 --output_dir output/test_valid_01 --train_batch_size 40 --num_train_epochs 300 --do_eval

# parameter:
#   data_dir: directory of training dataset
#   data_dir_for_val: directory of validation dataset
#   core_num: how many cores used to process data
#   output_dir: directory where model will be stored
#   train_batch_size: how may complex trained in one step
#   num_train_epochs: number of training epochs
#   do_eval

# train from background process
nohup python -u train.py --data_dir dataset/cv5_data/cv0/train --data_dir_for_val dataset/cv5_data/cv0/test --core_num 8 --output_dir output_cv5_id0 --train_batch_size 8 --eval_batch_size 8 --num_train_epochs 300 --do_eval --pwm_type pssm   >train_cv0501_01.log 2>&1 &

```

### 3.2 Train from pretrained model, also named five-tuning
```
python train.py --data_dir dataset/pri_data/train --data_dir_for_val dataset/pri_data/val --core_num 8 --output_dir output --train_batch_size 40 --num_train_epochs 100 --do_eval --resume --resume_path output/model.99.202.ckpt 

# parameter:
#   resume: set true to use resume model parameter
#   resume_path: directory of resume model parameter
```

### 3.2.1 Train from background, then we are not worry about terminal stoped
```
nohup python -u train.py --data_dir dataset/pri_data/train --data_dir_for_val dataset/pri_data/val --core_num 8 --output_dir output/fixmodel_01 --train_batch_size 32 --learning_rate 0.001 --num_train_epochs 300 --do_eval >train_fixmodel_01.log 2>&1 &

nohup python -u train.py --data_dir dataset/pri_data/train --data_dir_for_val dataset/pri_data/val --core_num 8 --output_dir output/test_valid_01 --train_batch_size 40 --learning_rate 0.001 --num_train_epochs 300 --do_eval >test_valid_01.log 2>&1 &

nohup python -u train.py --data_dir dataset/pri_data/train --data_dir_for_val dataset/pri_data/val --core_num 8 --output_dir /output/remodel_01 --train_batch_size 32 --learning_rate 0.001 --num_train_epochs 300 --do_eval --resume --resume_path output_02/model.300.252.ckpt   >train_02_02.log 2>&1 &

# will return pid, for example pid=2857798

# if we want to use chemistry input of DNA/RNA, we use --use_chemistry

nohup python -u train.py --data_dir dataset/pri_data/train --data_dir_for_val dataset/pri_data/val --core_num 8 --output_dir output_pssm_nc_04 --train_batch_size 32 --num_train_epochs 300 --do_eval --pwm_type pssm   >train_pssm_nc_04.log 2>&1 &

```

### 3.3 Inference
```
python train.py --do_test --data_dir_for_test dataset/pri_data/test --core_num 8 --output_dir output/fixmodel_01 --test_batch_size 16 --resume --resume_path output/fixmodel_01/model.300.252.ckpt


```


### 4. pretrain with other data
```

nohup python -u  train.py --data_name hox_data --data_dir dataset/hox_data/train --data_dir_for_val dataset/hox_data/val --core_num 8 --output_dir output_hox_01 --train_batch_size 280 --num_train_epochs 300 --do_eval  >train_hox_01.log 2>&1 &

```



```
### step1. put following files and directory under 'dataset/_datasets/cluster_res'
### cluster files:
├── protein_wt_cluster.tsv 
├── dsDNA_80_cluster.txt
├── dsRNA_80_cluster.txt
├── ssDNA_80_cluster.txt
├── ssRNA_80_cluster.txt
### main dataset:
├── seq_dg_v02.txt ### all
├── wt_v02.tsv ### w-type
### similarity files:
├── similarity
│   ├── dsdna_dist_230703.txt
│   ├── dsdna_dist_230710.txt
│   ├── dsrna_dist_230703.txt
│   ├── dsrna_dist_230710.txt
│   ├── protein_dist_230703.txt
│   ├── protein_dist_230710.txt
│   ├── ssdna_dist_230703.txt
│   ├── ssdna_dist_230710.txt
│   ├── ssrna_dist_230703.txt
│   └── ssrna_dist_230710.txt

### step2. edit data/data_cluster_analysis.py files:
## edit line 28 of "param_load = False", code will compute distance output file,
## if you set "param_load = True", code will try to read distance output file (if you do not have, error reported)
## edit line 31 of " output_dir = 'Your output dir' "

## then you can run:
$ python data/data_cluster_analysis.py
## then you can file train.csv and test.csv in 'Your output dir'
```