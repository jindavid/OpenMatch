CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task ranking \
        -model bert \
        -max_input 1280000 \
        -test /data2/jindawei/data/test_5193-1200_ANCE.jsonl \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./checkpoints/gltr_rank_n10_w1_step-1000 \
        -res ./results/gltr_rank_n100_test.trec \
        -max_query_len 32 \
        -max_doc_len 256 \
        -batch_size 32