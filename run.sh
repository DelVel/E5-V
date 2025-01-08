wandb online
HF_HOME=.cache/hf deepspeed ft_llm.py \
        llava-3b-cross-modal-contra-lr=4e-6-fp16 \
        \
        --lora.alpha 64 \
        --lora.dropout 0.05 \
        --lora.r 8 \
        --lora.target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
        \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --deepspeed ds.config \
        --learning_rate 4e-6 \
        --num_epochs 1 \
        \
        --save_steps 100
