CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task global_no_att \
        -model bert \
        -max_input 1280000 \
        -test /data2/jindawei/data/dev_valid_1200_subset_from-splitidx-0.top100.jsonl \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./checkpoints/gltr_global_n10_no_att_step-10000 \
        -res ./results/gltr_global_no_att_n100_test.trec \
        -max_query_len 32 \
        -max_doc_len 256 \
        -batch_size 32
