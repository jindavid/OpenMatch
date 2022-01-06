CUDA_VISIBLE_DEVICES=4 \
python inference.py \
        -task global \
        -model bert \
        -max_input 1280000 \
        -test /data2/jindawei/data/test_5193-1200_ANCE.jsonl \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./checkpoints/gltr_global_yes_CE_step-67500 \
        -res ./results/gltr_global_yes_CE_step-67500.trec \
        -max_query_len 32 \
        -max_doc_len 256 \
        -batch_size 32
