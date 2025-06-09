set -x
nproc_per_node=8
save_path="/tmp"
export HYDRA_FULL_ERROR=1

# Shift the arguments so $@ refers to the rest
# shift 2
echo $WORLD_SIZE $RANK 
# /cpfs/user/liuyanjiang/Eng/verl-dpskv2/data/sft_debug.parquet
# /cpfs/user/sunzekai/general_alignment/moe_sft/useful_moe_145b/moe_sft_145b_32k_v7.2.0_CIF/iter_0001000_hf
torchrun --nnodes=$WORLD_SIZE --nproc_per_node=$nproc_per_node --node_rank=$RANK  \
     -m moe_trainer.fsdp_sft_trainer \
    data.train_files=/cpfs/user/liuyanjiang/Eng/verl/examples/data/light-r1-stage1-v0.0.2.parquet \
    data.val_files=/cpfs/user/liuyanjiang/research/deepscaler/deepscaler/data/train/still.json \
    data.prompt_key=problem \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8192 \
    data.truncation=right \
    model.fsdp_config.model_dtype=bf16 \
    model.strategy=fsdp \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
    model.trust_remote_code=True \
    optim.warmup_steps_ratio=0.05 \
    optim.lr=5e-6 \
    optim.weight_decay=0.1 \
    model.enable_gradient_checkpointing=True \
    model.partial_pretrain=/cpfs/user/liuyanjiang/hf_models/moe_sft_145b_32k_v7-2-0_CIF_iter_0001589_hf \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=cybertron_sft_debug \
    trainer.total_epochs=4 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@