CUDA_VISIBLE_DEVICES= \
python sft_mlp.py \
    --model '' \
    --train_type lora \
    --dataset './datasets/numerical/train_quality.json, ./datasets/numerical/train_authenticity.json' \
    --val_dataset './datasets/numerical/eval_quality.json, ./datasets/numerical/eval_authenticity.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --freeze_vit True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --eval_steps  3200\
    --save_steps  3200\
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot

