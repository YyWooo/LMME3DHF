# Training
CUDA_VISIBLE_DEVICES= python main.py --dataset MINE --n_epochs 20 --lr_decoder 0.05 --lr_projector 0.1 --weight_decay 0.000001 --batch_size 1 --n_threads 8 --update_step 1 --manual_seed 42 --precision f32  --mm_train --image_path_MINE "database/image/" --salmap_path_MINE "database/sal_map/"

# Inference
CUDA_VISIBLE_DEVICES= python main.py --dataset MINE --model_name MINE_20250818_0109_SGD_d0.05_p0.1CAWR_ver19_s42_bs1Q_b0_full_mm  --n_threads 8 --batch_size 1 --no_train --no_val --test --resume_path "MINE_20250818_0109_SGD_d0.05_p0.1CAWR_ver19_s42_bs1Q_b0_full_mm/MINE_20250818_0109_SGD_d0.05_p0.1CAWR_ver19_s42_bs1Q_b0_full_mm_20.pth" --phase test  --manual_seed 42  --mm_train --precision f32
