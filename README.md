# VIT Tuner
Powerful and efficient VIT tuning tool: Integrating [Feature Distillation](https://github.com/SwinTransformer/Feature-Distillation), [Masked Autoencoder](https://github.com/facebookresearch/mae), and [FT-CLIP](https://github.com/LightDXY/FT-CLIP) techniques.

## Installation
1. **Use python version 3.10.0**
   ```bash
   pyenv local 3.10.0
   ```

2. **Create virtualenv & activate**
   ```bash
   python -m venv vit-tuner-env
   source vit-tuner-env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Command
1. **Train Model**
   ```bash
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
                         [--debug]
   ```
   1. **config_file**: Configuration file path. All configuration files are in configs/ directory.
   2. **exp_name**: Tag of the experiment. All experimental outputs, including model weights, training records will save at `checkpoints/{exp_name}` folder.
   3. **csv_path**: csv file that contains ground truth information. The file should contain ground truth column(s) and the column for spliting training and testing set.
   4. **image_dir**: Path to image directory.
   5. **n_epochs**: Number of epochs.
   6. **y_col**: Column name(s) corresponding to the ground truth(s).
   7. **bs**: Batch size.
   8. **lr**: Learning rate.
   9. **optim**: Type of optimizer. SHould be one of "Adam", "SGD", "AdamW".
   10. **weight_decay**: Weight decay for regularization.
   11. **weight**: MOdel weight path.
   12. **device**
   13. **num_workers** 

## TODO
1. Model Inference
2. Linear Probe
3. Model EMA