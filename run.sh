accelerate launch --config-file accel_train_config.json ft_llm.py \
        --lora.alpha 64 \
        --lora.dropout 0.05 \
        --lora.r 8 \
        --lora.target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
        \
        --per_device_train_batch_size 64 \
        --learning_rate 1e-4 \
        --num_epochs 1 \
        --resume_from_checkpoint False \
        --bf16 False \
        $@
