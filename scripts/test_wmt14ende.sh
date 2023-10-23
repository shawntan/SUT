# Average checkpoints
DATA=wmt14_en_de_joined_dict
CHECKPOINT_PATH=$1
export SPLIT=test

python3 $TRANSFORMER_CLINIC_ROOT/fairseq/scripts/average_checkpoints.py \
  --inputs $CHECKPOINT_PATH/checkpoint.best_nll_loss_*.pt \
  --output $CHECKPOINT_PATH/averaged_model.pt

# Evaluate
fairseq-generate \
    $TRANSFORMER_CLINIC_ROOT/data-bin/${DATA}\
    --path $CHECKPOINT_PATH/averaged_model.pt \
    --remove-bpe --lenpen 0.4 --beam 8 \
    --gen-subset $SPLIT \
    --user-dir . --results-path $CHECKPOINT_PATH/generate