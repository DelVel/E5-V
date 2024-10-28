RUN=e5v-8b-ours-relaxed-sigma-0.75

args=()

BASE_MODEL="models/llava-llama-3-8b"

R=64
ALPHA=16
BATCH_SIZE=640
MICRO_BATCH_SIZE=20
EPOCH=1
LR=4e-4

GPUS=8
NUM_NODES=4

wandb online

NCCL_DEBUG=ERROR deepspeed ft_llm.py \
        --base_model $BASE_MODEL \
        --output_dir $RUN \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --num_epochs $EPOCH \
        --learning_rate $LR \
        --lora_r $R \
        --lora_alpha $ALPHA \
        --deepspeed ds.config \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
        --bf16 true \
        ${args[@]}
