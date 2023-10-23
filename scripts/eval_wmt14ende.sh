#!/bin/bash
CHKPT_PATH=$1

export TRANSFORMER_CLINIC_ROOT=../Transformer-Clinic/
export SPLIT=test
./test_wmt14ende.sh $CHKPT_PATH && sleep 5 && (./get_ende_bleu.sh $CHKPT_PATH | tee $CHKPT_PATH/eval.log)
