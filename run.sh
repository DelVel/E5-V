RUN=e5v-8b

wandb online
NCCL_DEBUG=ERROR deepspeed ft_llm.py \
        --batch_size 384 \
        --data_path 'data/nli_for_simcse.csv' \
        --deepspeed ds.config \
        --learning_rate 4e-4 \
        --logging_steps 1 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_r 64 \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
        --micro_batch_size 96 \
        --num_epochs 2 \
        --output_dir $RUN \
        --save_steps 100
