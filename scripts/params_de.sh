export DATA=wmt14_en_de_joined_dict
export ARCHS=sut_transformer_base

export NUM_LAYERS=6
export EMBDIM=512
export ACC=1
export MAXTOKEN=8192

# MoA Parameters
export NUMEXPATT=48
export NUMATT=4
export ATTDIM=256
export HEADDIM=64

# MLP Parameters
export NUMEXPFF=48
export NUMFF=4
export FFDIM=512

export LR=7e-4
export DROPOUT=0.2
export ATTDROPOUT=0.2
export ACTDROPOUT=0.2
export HLTDROPOUT=0.0
export GATDROPOUT=0.2
export SAMPLETOPK=0
export CVLOSS=0
export SWITCHLOSS=0
export ZLOSS=0
export MILOSS=0.1
export ACTLOSS=0
export MAXEPOCH=100
export WARMUP=4000
export LOSS=sut_ce_loss
export WEIGHT_DECAY=0
