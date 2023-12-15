
###
```
1) put cluster into train/val/test
2) do seperately for each na type (na balance)
3) complex size balance
4) wt 80% in test
5) long to test
6) dg balance









nohup python -u train.py --data_dir dataset/tmp_data/train --data_dir_for_val dataset/tmp_data/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 40 --eval_batch_size 4 --learning_rate 0.0005 --num_train_epochs 300 --do_eval --pwm_type pssm >test_valid_02.log 2>&1 &
# bad, val loss not decrease



python train.py --debug --data_dir dataset/tmp_data/mid01/train --data_dir_for_val dataset/tmp_data/mid01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 32 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 300 --do_eval --pwm_type pssm --label_bin



nohup python -u train.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 40 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 300 --do_eval --pwm_type pssm --label_bin  >test_valid_02.log 2>&1 &


--sub_graph_depth 1


## use 18 bins
## use 8 bins with kmeans cluster

## use h=512


nohup python -u train.py --data_dir dataset/tmp_data/mid01/train --data_dir_for_val dataset/tmp_data/mid01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 16 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 300 --do_eval --pwm_type pssm --hidden_size 512 --label_bin  >test_valid_02.log 2>&1 &


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

nohup python -u train.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 600 --do_eval --pwm_type pssm --hidden_size 128 --label_bin  >test_valid_02.log 2>&1 &


nohup python -u train.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 600 --do_eval --hidden_size 128 --label_bin  >test_valid_02.log 2>&1 &


nohup python -u train.py --data_dir dataset/tmp_data/mid01/train --data_dir_for_val dataset/tmp_data/mid01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 600 --do_eval --hidden_size 128 --label_bin  >test_valid_02.log 2>&1 &


# change hard01 size

nohup python -u train.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 600 --do_eval --hidden_size 128  >test_valid_02.log 2>&1 &


nohup python -u train.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_03 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 600 --do_eval --hidden_size 128 --pwm_type pssm   >test_valid_03.log 2>&1 &


nohup python -u train.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 300 --do_eval --hidden_size 128 --use_prot_chm_feature  >test_valid_02.log 2>&1 &

### use simple datatset: 

validation epoch 0: loss = 1.2793, loss_logic = 0.0000, loss_reg = 1.2793, rvalue = 0.3969, pvalue = 0.0000, rrmse = 4.9252
validation epoch 1: loss = 1.1584, loss_logic = 0.0000, loss_reg = 1.1584, rvalue = 0.5280, pvalue = 0.0000, rrmse = 4.2390
validation epoch 2: loss = 1.0511, loss_logic = 0.0000, loss_reg = 1.0511, rvalue = 0.5945, pvalue = 0.0000, rrmse = 3.6885
validation epoch 3: loss = 0.9383, loss_logic = 0.0000, loss_reg = 0.9383, rvalue = 0.6472, pvalue = 0.0000, rrmse = 3.2017
validation epoch 4: loss = 0.9465, loss_logic = 0.0000, loss_reg = 0.9465, rvalue = 0.6488, pvalue = 0.0000, rrmse = 3.2358
validation epoch 5: loss = 0.9251, loss_logic = 0.0000, loss_reg = 0.9251, rvalue = 0.6566, pvalue = 0.0000, rrmse = 3.2355
validation epoch 6: loss = 0.9133, loss_logic = 0.0000, loss_reg = 0.9133, rvalue = 0.6801, pvalue = 0.0000, rrmse = 3.1410
validation epoch 7: loss = 0.9595, loss_logic = 0.0000, loss_reg = 0.9595, rvalue = 0.6493, pvalue = 0.0000, rrmse = 3.3756
validation epoch 8: loss = 0.8418, loss_logic = 0.0000, loss_reg = 0.8418, rvalue = 0.7010, pvalue = 0.0000, rrmse = 2.8887
validation epoch 9: loss = 0.8510, loss_logic = 0.0000, loss_reg = 0.8510, rvalue = 0.6934, pvalue = 0.0000, rrmse = 2.8841
validation epoch 10: loss = 0.8965, loss_logic = 0.0000, loss_reg = 0.8965, rvalue = 0.6979, pvalue = 0.0000, rrmse = 3.0782
validation epoch 11: loss = 0.8808, loss_logic = 0.0000, loss_reg = 0.8808, rvalue = 0.7094, pvalue = 0.0000, rrmse = 3.0234
validation epoch 12: loss = 0.8183, loss_logic = 0.0000, loss_reg = 0.8183, rvalue = 0.7188, pvalue = 0.0000, rrmse = 2.7236
validation epoch 13: loss = 0.8273, loss_logic = 0.0000, loss_reg = 0.8273, rvalue = 0.7146, pvalue = 0.0000, rrmse = 2.7771
validation epoch 14: loss = 0.8297, loss_logic = 0.0000, loss_reg = 0.8297, rvalue = 0.7072, pvalue = 0.0000, rrmse = 2.7786
validation epoch 15: loss = 0.8033, loss_logic = 0.0000, loss_reg = 0.8033, rvalue = 0.7239, pvalue = 0.0000, rrmse = 2.6387
validation epoch 16: loss = 0.8098, loss_logic = 0.0000, loss_reg = 0.8098, rvalue = 0.7193, pvalue = 0.0000, rrmse = 2.7358
validation epoch 17: loss = 0.8449, loss_logic = 0.0000, loss_reg = 0.8449, rvalue = 0.7037, pvalue = 0.0000, rrmse = 2.9172


nohup python -u train.py --data_dir dataset/tmp_data/simple/train --data_dir_for_val dataset/tmp_data/simple/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 100 --do_eval --hidden_size 128 --use_prot_chm_feature  >test_valid_02.log 2>&1 &


nohup python -u train.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 100 --do_eval --hidden_size 128 --use_prot_chm_feature  >test_valid_02.log 2>&1 &

nohup python -u train.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir output/test_valid_02 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 100 --do_eval --hidden_size 128 --use_prot_chm_feature  >test_valid_02.log 2>&1 &



Dataset/pri_data/dna_dataset/_dataset



nohup python -u train.py  --data_name hox_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir dna_dataset_task_01 --train_batch_size 48 --eval_batch_size 48 --learning_rate 0.001 --num_train_epochs 20 --do_eval --display_steps 250    >dna_dataset_task_01.log 2>&1 &


nohup python -u train.py  --data_name hox_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir dna_dataset_task_02 --train_batch_size 48 --eval_batch_size 48 --learning_rate 0.001 --num_train_epochs 20 --do_eval --display_steps 250 --use_prot_chm_feature    >dna_dataset_task_02.log 2>&1 &

python  train.py  --data_name hox_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir dna_dataset_debug --train_batch_size 48 --eval_batch_size 48 --learning_rate 0.001 --num_train_epochs 20 --do_eval --display_steps 1 --step_lr --steps_update_lr 5 --use_prot_chm_feature













nohup python -u train.py --data_dir dataset/tmp_data/simple/train --data_dir_for_val dataset/tmp_data/simple/val --core_num 8 --output_dir test_03 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 300 --do_eval --hidden_size 128 --use_prot_chm_feature --pwm_type hmm_pssm  >test_03.log 2>&1 &


nohup python -u train.py --data_dir dataset/tmp_data/simple/train --data_dir_for_val dataset/tmp_data/simple/val --core_num 8 --output_dir test_04 --train_batch_size 24 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 30 --do_eval --hidden_size 128 --use_prot_chm_feature --pwm_type hmm_pssm  >test_04.log 2>&1 &




## test orig with new arch
python train.py --debug --data_dir dataset/tmp_data/simple/train --data_dir_for_val dataset/tmp_data/simple/val --core_num 8 --output_dir debug --train_batch_size 8 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 30 --do_eval --hidden_size 1024 --use_prot_chm_feature --pwm_type hmm_pssm

## test deep emb with new arch
python train.py --data_dir dataset/tmp_data/simple/train --data_dir_for_val dataset/tmp_data/simple/val --core_num 8 --output_dir debug --train_batch_size 8 --eval_batch_size 4 --learning_rate 0.001 --num_train_epochs 30 --do_eval --hidden_size 1024 --use_deep_emb --prot_emb_size 2560 --nc_emb_size 768



## run deep emb with new arch
##
nohup python -u train.py --data_dir dataset/tmp_data/simple/train --data_dir_for_val dataset/tmp_data/simple/val --core_num 8 --output_dir deep_emb_01 --train_batch_size 8 --eval_batch_size 4 --learning_rate 0.0001 --num_train_epochs 50 --do_eval --hidden_size 1024 --use_deep_emb --prot_emb_size 2560 --nc_emb_size 768  >deep_emb_01.log 2>&1 &

epoch = 7 step = 750, loss = 0.5202, loss_logic = 0.0000, loss_reg = 0.5202, time = 0.10, lr = 0.00010
Iter (loss=X.XXX): 100%|██████████| 542/542 [00:11<00:00, 46.25it/s]
validation epoch 7: loss = 0.8114, loss_logic = 0.0000, loss_reg = 0.8114, rvalue = 0.7136, pvalue = 0.0000, rrmse = 2.6378

## --use_repeat_sampler

##
nohup python -u train.py --data_dir dataset/tmp_data/simple/train --data_dir_for_val dataset/tmp_data/simple/val --core_num 8 --output_dir deep_emb_04 --train_batch_size 16 --eval_batch_size 4 --learning_rate 0.0001 --num_train_epochs 200 --do_eval --hidden_size 1024 --use_deep_emb --prot_emb_size 2560 --nc_emb_size 768 --warmup_epoch 30 --direct_read_cache >deep_emb_04.log 2>&1 &


nohup python -u train.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir deep_emb_05 --train_batch_size 16 --eval_batch_size 4 --learning_rate 0.00001 --num_train_epochs 300 --do_eval --hidden_size 1024 --use_deep_emb --prot_emb_size 2560 --nc_emb_size 768 --warmup_epoch 20 --direct_read_cache --use_repeat_sampler >deep_emb_05.log 2>&1 &

