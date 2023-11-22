#### dep packages
```

```

#### record
```



CUDA_VISIBLE_DEVICES=0
nohup

## 01
  lr = 1e-4

CUDA_VISIBLE_DEVICES=0   nohup python -u    train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_01 --train_batch_size 8 --eval_batch_size 8 --learning_rate 0.0001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch   >ddata_pretrain_01.log 2>&1 &

## 02
  lr = 1e-5

CUDA_VISIBLE_DEVICES=0   nohup python -u    train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_02 --train_batch_size 8 --eval_batch_size 8 --learning_rate 0.00001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch >ddata_pretrain_02.log 2>&1 &


## 03
  + mask both input, lr = 1e-5
  process be killed when server re-started
  
CUDA_VISIBLE_DEVICES=0   nohup python -u    train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 2 --output_dir ddata_pretrain_03 --train_batch_size 10 --eval_batch_size 10 --learning_rate 0.00001 --num_train_epochs 10 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch --step_lr  >ddata_pretrain_03.log 2>&1 &

## 03B
  + mask both input, lr = 1e-5, step_update_lr = 5000

CUDA_VISIBLE_DEVICES=0   nohup python -u    train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 2 --output_dir ddata_pretrain_03 --train_batch_size 10 --eval_batch_size 10 --learning_rate 0.00001 --num_train_epochs 10 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch --step_lr  --resume --resume_path output/ddata_pretrain_03/checkpoint/tmp.ckpt  >ddata_pretrain_03B.log 2>&1 &

## 03B
  + mask both input, lr = 5e-5, step_update_lr = 50000 

CUDA_VISIBLE_DEVICES=0 nohup python -u train_esm.py \
--dataset_type ext \
--data_name dna_data \
--data_dir dataset/dna_dataset/_dataset/train \
--data_dir_for_val dataset/dna_dataset/_dataset/val \
--output_dir ddata_pretrain_04 \
--do_eval \
--train_batch_size 10 \
--eval_batch_size 10 \
--num_train_epochs 5 \
--learning_rate 0.00005 \
--hidden_size 1024 \
--freeze_layer -1 \
--display_steps 100 \
--big_dataset \
--save_model_epoch \
--step_lr \
--steps_update_lr 50000 \
>ddata_pretrain_04.log 2>&1 &
```