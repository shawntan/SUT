#!/bin/bash
CHECKPOINT_PATH=$1
DATA=wmt14_en_de_joined_dict
SPLIT=test

python clean_generate.py --generate_file $CHECKPOINT_PATH/generate/generate-$SPLIT.txt --output_file $CHECKPOINT_PATH/generate/generate-$SPLIT.cleaned

mosesdecoder=$TRANSFORMER_CLINIC_ROOT/pre-process/mosesdecoder
# tok_gold_targets=$TRANSFORMER_CLINIC_ROOT/pre-process/wmt14_en_de/tmp/test.de

tok_gold_targets=$TRANSFORMER_CLINIC_ROOT/pre-process/wmt14_en_de/tmp/$SPLIT.de

decodes_file=$CHECKPOINT_PATH/generate/generate-${SPLIT}.cleaned
ls $decodes_file

# Replace unicode.
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $decodes_file > $decodes_file.n
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $tok_gold_targets > $decodes_file.gold.n

# # Tokenize.
# perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decodes_file.n > $decodes_file.tok
# perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decodes_file.gold.n > $decodes_file.gold.tok

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.gold.n > $decodes_file.gold.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.n > $decodes_file.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $decodes_file.gold.atat < $decodes_file.atat

# Detokenize.
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l de < $decodes_file.n > $decodes_file.detok
perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l de < $decodes_file.gold.n > $decodes_file.gold.detok

sacrebleu $decodes_file.gold.detok -l en-de -i $decodes_file.detok -b
