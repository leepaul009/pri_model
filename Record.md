

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

# to check if layer-init method lead to bad performance?

nohup python -u train.py --data_dir data/train --data_dir_for_val data/val --core_num 8 --output_dir output_pssm_nc_03 --train_batch_size 32 --num_train_epochs 300 --do_eval --pwm_type pssm   >train_pssm_nc_03.log 2>&1 &


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












































