# pri_model


## setup conda environment
find deps.yml file inside repo, and create conda environment.
```
### import conda env from yml file
conda env create -f deps.yml
```

## dataset
dataset is stored in the directory:
```
data/train/
data/val/
data/test/
```

# train from beginning
```
# data_dir: directory of dataset
# core_num: use 8 cores to process data
# output_dir: model stored here
python train.py --data_dir data --core_num 8 --output_dir output --train_batch_size 8
```



