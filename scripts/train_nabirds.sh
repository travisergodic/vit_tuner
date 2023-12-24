config_file=./configs/feautre_distillation/_fd_vit_base.py
exp_name=fd_vit_base_v1
csv_path=/content/nabirds.csv
image_dir=/content/hpdh/nabirds/images
split_col=is_train
n_epochs=100
y_col=label
bs=64
lr=0.0001
optim=AdamW
weight_decay=0.05
device=cuda
num_workers=2


python tools/train.py --config_file ${config_file} \
                      --exp_name ${exp_name} \
                      --csv_path ${csv_path} \
                      --image_dir ${image_dir} \
                      --split_col ${split_col} \
                      --n_epochs ${n_epochs} \
                      --y_col ${y_col} \
                      --bs ${bs} \
                      --lr ${lr} \
                      --optim ${optim} \
                      --weight_decay ${weight_decay} \
                      --weight ${weight} \
                      --device ${device} \
                      --num_workers ${num_workers} \
                      --debug