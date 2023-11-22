

## new model

# round 1:
    lr = 1e-5, lr_sche=exp-decay
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_01 --train_batch_size 2 --eval_batch_size 2 --learning_rate 0.00001 --num_train_epochs 300 --do_eval --hidden_size 128 --warmup_epoch 20 --display_steps 100 >new_model_01.log 2>&1 &

# round 2: 
    update nc pooling(out nc's C=768/2 similar to prot's C)
    lr = 1e-4, lr_sche=cosine
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_02 --train_batch_size 2 --eval_batch_size 2 --learning_rate 0.0001 --num_train_epochs 300 --do_eval --hidden_size 128 --warmup_epoch 20 --display_steps 100 >new_model_02.log 2>&1 &

# round 3: 
    hard test dataset
    update nc pooling(out nc's C=768/2 similar to prot's C)
    lr = 1e-4, lr_sche=cosine
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir new_model_03 --train_batch_size 2 --eval_batch_size 2 --learning_rate 0.0001 --num_train_epochs 300 --do_eval --hidden_size 128 --warmup_epoch 20 --display_steps 100 >new_model_03.log 2>&1 &

# round 4: 
    hard test dataset
    update nc pooling(out nc's C=768/2 similar to prot's C)
    lr = 1e-5, lr_sche=cosine
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir new_model_04 --train_batch_size 2 --eval_batch_size 2 --learning_rate 0.00001 --num_train_epochs 120 --do_eval --hidden_size 128 --warmup_epoch 10 --display_steps 100 >new_model_04.log 2>&1 &


# round 5: 
    update nc pooling(out nc's C=768/2 similar to prot's C)
    lr = 1e-5, lr_sche=cosine
    bert mask
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_05 --train_batch_size 2 --eval_batch_size 2 --learning_rate 0.00001 --num_train_epochs 300 --do_eval --hidden_size 128 --warmup_epoch 20 --display_steps 100 >new_model_05.log 2>&1 &


# round 6: 
    lr = 5e-4, lr_sche=cosine
    bert mask
    inter-net
    freeze first 2 tf layers
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_06 --train_batch_size 4 --eval_batch_size 4 --learning_rate 0.0005 --num_train_epochs 150 --do_eval --hidden_size 128 --freeze_layer 1 --warmup_epoch 10 --display_steps 100 >new_model_06.log 2>&1 &

at epoch 33, loss = 0.9888, rvalue = 0.5968, rrmse = 3.3476

# round 7: 
    lr = 5e-5, lr_sche=cosine
    bert mask
    inter-net
    freeze first 3 tf layers

nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_07 --train_batch_size 4 --eval_batch_size 4 --learning_rate 0.00005 --weight_decay 0.03 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer 2 --warmup_epoch 5 --display_steps 100 >new_model_07.log 2>&1 &


# round 8: 
    lr = 1e-5, lr_sche=cosine
    - bert mask
    inter-net
    do not freeze

nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_08 --train_batch_size 2 --eval_batch_size 2 --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_08.log 2>&1 &

at epoch 130, loss = 0.9218, rvalue = 0.6068, rrmse = 3.172

# round 9: 
    lr = 1e-5, lr_sche=cosine
    - bert mask
    inter-net
    do not freeze
    use_repeat_sampler
# get best performance:
    dataset =  hard02
    at epoch 237, loss = 0.859, rvalue = 0.6711, rrmse = 3.0079

nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_09 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_09.log 2>&1 &



# round 10: 

nohup python -u train_esm.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir new_model_10 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_10.log 2>&1 &


# round 11: 
    + mask tokens
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard01/train --data_dir_for_val dataset/tmp_data/hard01/val --core_num 8 --output_dir new_model_11 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_11.log 2>&1 &


# round 12: 
    layer 5, dist-sample=0.5
    at epoch 51, loss = 0.8517, rvalue = 0.6657, rrmse = 2.8475

nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_12 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 400 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_12.log 2>&1 &


# round 13: 
    layer 4, dist-sample=0.5
    bad, at epoch 229, loss = 0.9911, rvalue = 0.6085, rrmse = 3.8811

nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_13 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 400 --do_eval --hidden_size 512 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_13.log 2>&1 &

# round 14: 
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5
    at epoch 163, loss = 2.907, rvalue = 0.6772, rrmse = 2.907
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_14 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 400 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_14.log 2>&1 &

# round 15: 
    layer 6, dist-sample=0.5, MSELoss, lr = 1e-5
    at epoch 28, loss = 3.0588, rvalue = 0.6653, rrmse = 3.0588
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_15 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 400 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_15.log 2>&1 &

# round 16: 
    layer 6, dist-sample=0.5, MSELoss, lr = 7e-6
    at epoch 18, loss = 3.3564, rvalue = 0.6656, rrmse = 3.3564
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_16 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.000007 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_16.log 2>&1 &

# round 17: 
    layer 6, dist-sample=0.5, MSELoss, lr = 1e-6
    at epoch 46, loss = 3.0907, rvalue = 0.6742, rrmse = 3.0907
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_17 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.000001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 >new_model_17.log 2>&1 &


############ train with pretrain params
# round 18:
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-6
    use pretrain model from ddataset
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_18 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.000001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 --resume --resume_path dataset/checkpoints/pretrain_0906.ckpt  >new_model_18.log 2>&1 &

# round 19:
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-6
    use pretrain model from ddataset
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_19 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.000001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 --resume --resume_path dataset/checkpoints/pretrain_0906_no_pred_head.ckpt  >new_model_19.log 2>&1 &

# round 20:
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5
    use 1e pretrain model from ddataset
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_20 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 --resume --resume_path dataset/checkpoints/pretrain_0906_no_pred_head.ckpt  >new_model_20.log 2>&1 &

at epoch 245, valloss = 3.1439, rvalue = 0.6591, rrmse = 3.1439

# round 21:
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5
    use 2e pretrain model from ddataset
nohup python -u train_esm.py --data_dir dataset/tmp_data/hard02/train --data_dir_for_val dataset/tmp_data/hard02/val --core_num 8 --output_dir new_model_21 --train_batch_size 2 --eval_batch_size 2 --use_repeat_sampler --learning_rate 0.00001 --weight_decay 0.02 --num_train_epochs 300 --do_eval --hidden_size 1024 --freeze_layer -1 --warmup_epoch 5 --display_steps 100 --resume --resume_path dataset/checkpoints/pretrain_e2_431m_no_pred_head.ckpt  >new_model_21.log 2>&1 &

at epoch 219, valloss = 3.097, rvalue = 0.648, rrmse = 3.097

# round 22:
    layer 5, dist-sample=0.5, MSELoss, lr = 7e-5
    use 2e pretrain model from ddataset
nohup python -u train_esm.py \
--data_dir dataset/tmp_data/hard02/train \
--data_dir_for_val dataset/tmp_data/hard02/val \
--output_dir new_model_22 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 300 \
--use_repeat_sampler \
--learning_rate 0.00007 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
--resume \
--resume_path dataset/checkpoints/pretrain_e2_431m_no_pred_head.ckpt \
>new_model_22.log 2>&1 &

at epoch 54, valloss = 3.1609, rvalue = 0.6373, rrmse = 3.1609

# round 23:
    layer 5, dist-sample=0.5, MSELoss, lr = 2e-5, epoch 400
    use 2e pretrain model from ddataset
nohup python -u train_esm.py \
--data_dir dataset/tmp_data/hard02/train \
--data_dir_for_val dataset/tmp_data/hard02/val \
--output_dir new_model_23 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 400 \
--use_repeat_sampler \
--learning_rate 0.00002 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
--resume \
--resume_path dataset/checkpoints/pretrain_e2_431m_no_pred_head.ckpt \
>new_model_23.log 2>&1 &

at epoch 153, valloss = 3.1482, rvalue = 0.6368, rrmse = 3.1482

# round ?:
    pretrain
CUDA_VISIBLE_DEVICES=0 nohup python -u train_esm.py \
--dataset_type ext \
--data_name dna_data \
--data_dir dataset/dna_dataset/_dataset/train \
--data_dir_for_val dataset/dna_dataset/_dataset/val \
--output_dir dd_ptrain_01 \
--do_eval \
--train_batch_size 10 \
--eval_batch_size 10 \
--num_train_epochs 5 \
--learning_rate 0.0001 \
--hidden_size 1024 \
--freeze_layer -1 \
--display_steps 100 \
--big_dataset \
--save_model_epoch \
--step_lr \
--step_lr_warmup \
--steps_update_lr 50000 \
--resume \
--resume_path dataset/checkpoints/pretrain_esm_dbert_layer5_mse.ckpt \
>output/dd_ptrain_01.log 2>&1 &


############ train on dna-only-dataset
# round 24:
    dataset with only dna
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5, epoch 300
    at epoch 148, valloss = 2.7674, rvalue = 0.6809, rrmse = 2.7674
nohup python -u train_esm.py \
--data_dir dataset/tmp_data/dna_hard02/train \
--data_dir_for_val dataset/tmp_data/dna_hard02/val \
--output_dir nm_dna_24 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 300 \
--learning_rate 0.00001 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
>nm_dna_24.log 2>&1 &

at epoch 148, valloss = 2.7674, rvalue = 0.6809, rrmse = 2.7674

# round 25:
    dataset with only dna
    layer 5, dist-sample=0.5, MSELoss, epoch 300
    lr = 8e-6(bad 0.66,overfit)
    lr = 1e-6(bad 0.61)
    lr = 1e-6()
    update interval of repeat-sampler
    set cos_lr_schedular lr_min = lr*1e-3
nohup python -u train_esm.py \
--data_dir dataset/tmp_data/dna_hard02/train \
--data_dir_for_val dataset/tmp_data/dna_hard02/val \
--output_dir nm_dna_25 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--use_repeat_sampler \
--num_train_epochs 400 \
--learning_rate 0.000001 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
>nm_dna_25.log 2>&1 &


# round 26:
    dataset with only dna
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5, epoch 300
    at epoch 49, valloss = 2.9923, rvalue = 0.7371, rrmse = 2.9923

nohup python -u train_esm.py \
--data_dir dataset/tmp_data/dna_hard03/train \
--data_dir_for_val dataset/tmp_data/dna_hard03/val \
--output_dir nm_dna_26 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 300 \
--learning_rate 0.00001 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
>output/nm_dna_26.log 2>&1 &


# round 27:
    dataset with only dna
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5, epoch 300, lr-sched=exp-decay

nohup python -u train_esm.py \
--data_dir dataset/tmp_data/dna_hard02B/train \
--data_dir_for_val dataset/tmp_data/dna_hard02B/val \
--output_dir nm_dna_27 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 300 \
--learning_rate 0.00001 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
>output/nm_dna_27.log 2>&1 &


# round 28:
    dataset with only dna
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5, epoch 150, lr-sched=cos(end_lr=0)
    at epoch 51, valloss = 3.0426, rvalue = 0.7252, rrmse = 3.0426 
nohup python -u train_esm.py \
--data_dir dataset/tmp_data/dna_hard03/train \
--data_dir_for_val dataset/tmp_data/dna_hard03/val \
--output_dir nm_dna_28 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 150 \
--learning_rate 0.00001 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
>output/nm_dna_28.log 2>&1 &


# round 29:
    dataset with only dna
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5, epoch 300, lr-sched=cos
    at epoch 163, valloss = 2.749, rvalue = 0.6884, rrmse = 2.749

nohup python -u train_esm.py \
--data_dir dataset/tmp_data/dna_hard02/train \
--data_dir_for_val dataset/tmp_data/dna_hard02/val \
--output_dir nm_dna_29 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 300 \
--learning_rate 0.00001 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
>output/nm_dna_29.log 2>&1 &

# round 30:
    dataset with only dna
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5, epoch 300, lr-sched=cos(T_max = 50)
    at epoch 201, valloss = 2.8591, rvalue = 0.6639, rrmse = 2.8591
    
nohup python -u train_esm.py \
--data_dir dataset/tmp_data/dna_hard02/train \
--data_dir_for_val dataset/tmp_data/dna_hard02/val \
--output_dir nm_dna_30 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 300 \
--learning_rate 0.00001 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
>output/nm_dna_30.log 2>&1 &


# round 31:
    dataset with only dna
    layer 5, dist-sample=0.5, MSELoss, lr = 1e-5, epoch 300, lr-sched=cos(T_max = 50)
    at epoch 116, valloss = 3.4223, rvalue = 0.5771, rrmse = 3.4223
    
nohup python -u train_esm.py \
--data_dir dataset/tmp_data/dna_hard02/train \
--data_dir_for_val dataset/tmp_data/dna_hard02/val \
--output_dir nm_dna_31 \
--core_num 8 \
--do_eval \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_train_epochs 300 \
--learning_rate 0.00001 \
--weight_decay 0.02 \
--hidden_size 1024 \
--freeze_layer -1 \
--warmup_epoch 5 \
--display_steps 100 \
--resume \
--resume_path dataset/checkpoints/tmp_1017_no_pred_head.ckpt \
>output/nm_dna_31.log 2>&1 &
```





























