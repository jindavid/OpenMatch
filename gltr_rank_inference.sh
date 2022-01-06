CUDA_VISIBLE_DEVICES=3 \
python inference.py \
        -task ranking \
        -model bert \
        -max_input 1280000 \
        -test /data2/jindawei/data/test_5193-1200_ANCE.jsonl \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./checkpoints/gltr_rank_om_margin_step-45000 \
        -res ./results/gltr_rank_om_margin_step-45000_test.trec \
        -max_query_len 32 \
        -max_doc_len 256 \
        -batch_size 8