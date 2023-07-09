

### record 2023/06/17
```
## morning
# use chemistry input
# use pssm pwm for protein
# use same-weight-prot-nc-sequence-concat-feature
# use detectron2's method to init all nn.Conv2d
# use detectron2's method to init all TimeDistributed and Dense

notice: same-weight-prot-nc-sequence-concat-feature: first use maxpool for prot and nc seprately, then concat prot and nc whole-feature

nohup python -u train.py --data_dir data/train --data_dir_for_val data/val --core_num 8 --output_dir output_pssm_nc_03 --train_batch_size 32 --num_train_epochs 300 --do_eval --pwm_type pssm --use_chemistry   >train_pssm_nc_03.log 2>&1 &
# loss do not decrease very well, 
# need to check if chemistry input, or layer-init method lead to bad performance?




## night
# use pssm pwm for protein
# use same-weight-prot-nc-sequence-concat-feature
# use detectron2's method to init all nn.Conv2d
# use detectron2's method to init all TimeDistributed and Dense

# to check if layer-init method lead to bad performance? good

nohup python -u train.py --data_dir data/pri_data/train --data_dir_for_val data/pri_data/val --core_num 8 --output_dir output_pssm_nc_03 --train_batch_size 32 --num_train_epochs 300 --do_eval --pwm_type pssm   >train_pssm_nc_03.log 2>&1 &

# last epoch:
# validation loss = 0.4110, rvalue = 0.8597, pvalue = 0.00000000, rrmse = 1.3221

# do test
python train.py --do_test --data_dir_for_test data/pri_data/test --core_num 8 --output_dir output --test_batch_size 16 --resume --resume_path output/output_pssm_nc_03/model.274.252.ckpt --pwm_type pssm

# test result:
Test loss = 0.5421, rvalue = 0.7878, pvalue = 0.00000000, rrmse = 1.8332
gts: [-0.93 -0.98 -0.98 ... -9.5  -9.48 -8.78] 
preds: [-1.5985017 -1.5537763 -1.5537763 ... -9.614284  -9.7450695 -9.732097 ] 

```


### record 2023/06/18
```
## morning
# use chemistry input
# use pssm pwm for protein
# use same-weight-prot-nc-sequence-concat-feature
# use detectron2's method to init all nn.Conv2d
# use detectron2's method to init all TimeDistributed and Dense

nohup python -u train.py --data_dir data/train --data_dir_for_val data/val --core_num 8 --output_dir output_pssm_nc_04 --train_batch_size 32 --num_train_epochs 300 --do_eval --pwm_type pssm --use_chemistry   >train_pssm_nc_04.log 2>&1 &
## loss do not decrease

```

### record 2023/06/19
```
```



### summary
```
1)
what is updated:
  - use hmm pwm as input
300 epoch's validation rvalue = 0.87
test rvalue = 0.6?

2)
what is updated:
  - use pssm pwm as input
300 epoch's validation rvalue = 0.87
test rvalue = 0.75

3）
what is updated: 
  - do maxpooling to protein and DNA/RNA embedding seprately
  - use pssm pwm as input
validation rvalue = 0.86
test rvalue = 0.78

4）
what is updated: 
  - do maxpooling to protein and DNA/RNA embedding seprately
  - use pssm pwm as input
  - DNA/RNA chemistry feature
performance is very bad, loss can not decrease during training of the deeplearning-model

5）
pretrain with table_hox_zscore_102107 dataset(very big dataset), face a memory error (I am fixing now)
```


### record 2023/06/29
```
nohup python -u train.py --data_dir data/cv5_data/cv0/train --data_dir_for_val data/cv5_data/cv0/test --core_num 8 --output_dir output/output_cv5_id0 --train_batch_size 8 --eval_batch_size 4 --num_train_epochs 300 --do_eval --pwm_type pssm   >train_cv0501_01.log 2>&1 &



nohup python -u train_cv.py --data_dir data/cv5_data/cv0/train --data_dir_for_val data/cv5_data/cv0/test --core_num 8 --output_dir output/output_cv5_id0 --train_batch_size 8 --eval_batch_size 8 --num_train_epochs 300 --do_eval --pwm_type pssm  >train_cv0501_01.log 2>&1 &

```



###
```
1) put cluster into train/val/test
2) do seperately for each na type (na balance)
3) complex size balance
4) wt 80% in test
5) long to test
6) dg balance









nohup python -u train.py --data_dir data/tmp_data/train --data_dir_for_val data/tmp_data/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 40 --eval_batch_size 4 --learning_rate 0.0005 --num_train_epochs 300 --do_eval --pwm_type pssm >test_valid_02.log 2>&1 &
# bad, val loss not decrease



python train.py --debug --data_dir data/tmp_data/mid01/train --data_dir_for_val data/tmp_data/mid01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 32 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 300 --do_eval --pwm_type pssm --label_bin



nohup python -u train.py --data_dir data/tmp_data/hard01/train --data_dir_for_val data/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 40 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 300 --do_eval --pwm_type pssm --label_bin  >test_valid_02.log 2>&1 &


--sub_graph_depth 1


## use 18 bins
## use 8 bins with kmeans cluster
## use h=512


nohup python -u train.py --data_dir data/tmp_data/mid01/train --data_dir_for_val data/tmp_data/mid01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 16 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 300 --do_eval --pwm_type pssm --hidden_size 512 --label_bin  >test_valid_02.log 2>&1 &


## use hard dataset
## use 8 bins with kmeans cluster
## use h=256
## use dropout in sub/global graph, p=0.1
## use cos lr sched
### bad

## use hard dataset
## use 8 bins with kmeans cluster
## use h=256
## use dropout in sub/global graph, p=0.1 (global slightly diff)
## use cos lr sched, min_lr=def
## use 3 depths for global graph
### ing

nohup python -u train.py --data_dir data/tmp_data/hard01/train --data_dir_for_val data/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 32 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 600 --do_eval --pwm_type pssm --hidden_size 128 --label_bin  >test_valid_02.log 2>&1 &

```





























