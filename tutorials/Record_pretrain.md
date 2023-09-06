#### dep packages
```

```

#### record
```






nohup python -u train_esm.py --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_01 --train_batch_size 48 --eval_batch_size 48 --learning_rate 0.0001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --save_model_epoch  >ddata_pretrain_01.log 2>&1 &


python train_esm.py --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_01 --train_batch_size 48 --eval_batch_size 48 --learning_rate 0.0001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --save_model_epoch --direct_read_cache --tmp_dir output/ddata_pretrain_01/tmp_data

~~~~~~~~~~~~~~

python   train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_01 --train_batch_size 8 --eval_batch_size 8 --learning_rate 0.0001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch


CUDA_VISIBLE_DEVICES=0
nohup

CUDA_VISIBLE_DEVICES=0   nohup python -u    train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_01 --train_batch_size 8 --eval_batch_size 8 --learning_rate 0.0001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch   >ddata_pretrain_01.log 2>&1 &


CUDA_VISIBLE_DEVICES=0   nohup python -u    train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_02 --train_batch_size 8 --eval_batch_size 8 --learning_rate 0.00001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch >ddata_pretrain_02.log 2>&1 &


--direct_read_cache --tmp_dir dataset/dna_dataset/tmp_data


nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_14 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 400 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_14.log 2>&1 &

## 03
  + mask both input
  
CUDA_VISIBLE_DEVICES=0   nohup python -u    train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 2 --output_dir ddata_pretrain_03 --train_batch_size 10 --eval_batch_size 10 --learning_rate 0.00001 --num_train_epochs 10 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch --step_lr  >ddata_pretrain_03.log 2>&1 &

## 03B
  + mask both input

CUDA_VISIBLE_DEVICES=0   nohup python -u    train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 2 --output_dir ddata_pretrain_03 --train_batch_size 10 --eval_batch_size 10 --learning_rate 0.00001 --num_train_epochs 10 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch --step_lr --resume --resume_path output/ddata_pretrain_03/checkpoint/tmp.ckpt  >ddata_pretrain_03B.log 2>&1
```