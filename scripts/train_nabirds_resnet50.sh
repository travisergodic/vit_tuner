config_file=./configs/_resnet50_v1_single.py
exp_name=resnet50_v1
csv_path=/content/nabirds.csv
image_dir=/content/hpdh/nabirds/images
split_col=split
n_epochs=100
y_col=label
bs=64
lr=0.0001
optim=AdamW
weight_decay=0.05
device=cuda
num_workers=2
eval_freq=3


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
                      --device ${device} \
                      --num_workers ${num_workers} \
                      --eval_freq ${eval_freq}