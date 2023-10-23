#!/bin/bash
export PYTHONFAULTHANDLER=1

# Using total $WORLD_SIZE * $NUM_GPU = 32
export WORLD_SIZE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export NUM_GPU=2

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=51234

export OUTPUT_DIR=checkpoints/$EXP_NAME
export LOG_DIR=logs/$EXP_NAME
export RANK=$SLURM_PROCID

echo "EXP_NAME" $EXP_NAME
echo "OUTPUT_DIR" $OUTPUT_DIR
echo "Distributed training:"
echo MASTER_ADDR $MASTER_ADDR
echo MASTER_PORT $MASTER_PORT
echo RANK $RANK

export OMP_NUM_THREADS=4

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPU \
    --nnodes=$WORLD_SIZE:$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$((MASTER_PORT+1)) --rdzv_id ${EXP_NAME} \
    --max_restarts=10 \
    $(which fairseq-train) ../Transformer-Clinic/data-bin/${DATA} \
    --arch ${ARCHS} --share-all-embeddings --amp \
    --encoder-halting --decoder-halting --actloss ${ACTLOSS} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 --clip-norm 0.0 \
    --warmup-init-lr 1e-07 --lr ${LR} --lr-scheduler inverse_sqrt --warmup-updates ${WARMUP} \
    --encoder-embed-dim ${EMBDIM} --decoder-embed-dim ${EMBDIM} \
    --encoder-attn-num-expert ${NUMEXPATT}  --encoder-attn-k ${NUMATT} --encoder-attn-expert-dim ${ATTDIM} \
    --decoder-attn-num-expert ${NUMEXPATT}  --decoder-attn-k ${NUMATT} --decoder-attn-expert-dim ${ATTDIM} \
    --encoder-ff-num-expert ${NUMEXPFF} --encoder-ff-k ${NUMFF} --encoder-ff-expert-dim ${FFDIM} \
    --decoder-ff-num-expert ${NUMEXPFF} --decoder-ff-k ${NUMFF} --decoder-ff-expert-dim ${FFDIM} \
    --dropout ${DROPOUT} --attention-dropout ${ATTDROPOUT} --activation-dropout ${ACTDROPOUT} \
    --halting-dropout ${HLTDROPOUT} --gating-dropout ${GATDROPOUT} --weight-decay ${WEIGHT_DECAY} \
    --encoder-layers ${NUM_LAYERS} \
    --head-dim ${HEADDIM} --decoder-layers ${NUM_LAYERS} \
    --sample-topk ${SAMPLETOPK} --cvloss ${CVLOSS} --switchloss ${SWITCHLOSS} --zloss ${ZLOSS} --miloss ${MILOSS} \
    --criterion $LOSS --label-smoothing 0.1 \
    --tensorboard-logdir ${OUTPUT_DIR} \
    --update-freq ${ACC} \
    --user-dir . \
    --max-tokens ${MAXTOKEN} --max-epoch ${MAXEPOCH} \
    --save-dir ${OUTPUT_DIR} --continue-once 1 \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric nll_loss \
    --keep-last-epochs 10 >> $LOG_DIR/$RANK.log

