#### dep packages
```

```

#### record
```






nohup python -u train_esm.py --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_01 --train_batch_size 48 --eval_batch_size 48 --learning_rate 0.0001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --save_model_epoch  >ddata_pretrain_01.log 2>&1 &


python train_esm.py --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_01 --train_batch_size 48 --eval_batch_size 48 --learning_rate 0.0001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --save_model_epoch --direct_read_cache --tmp_dir output/ddata_pretrain_01/tmp_data


python train_esm.py --dataset_type ext --data_name dna_data --data_dir dataset/dna_dataset/_dataset/train --data_dir_for_val dataset/dna_dataset/_dataset/val --core_num 8 --output_dir ddata_pretrain_01 --train_batch_size 48 --eval_batch_size 48 --learning_rate 0.0001 --num_train_epochs 20 --do_eval --hidden_size 1024 --freeze_layer -1 --display_steps 100 --big_dataset --save_model_epoch 




--direct_read_cache --tmp_dir dataset/dna_dataset/tmp_data





```