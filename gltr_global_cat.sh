CUDA_VISIBLE_DEVICES=4 \
python train.py \
        -task global_cat \
        -model bert \
        -train /data2/jindawei/data/msmarco_train_global_cat_top98_ANCE+BM25_10000.json \
        -max_input 40000000 \
        -save ./checkpoints/gltr_global_cat1 \
        -dev /data2/jindawei/data/dev_valid_1200_subset_from-splitidx-0.top100.jsonl \
        -qrels /data2/jindawei/data/msmarco-docdev-qrels.tsv \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -res ./results/gltr_global_cat1.trec \
        -metric mrr_cut_100 \
        -max_query_len 64 \
        -max_doc_len 445 \
        -epoch 1 \
        -optimizer adamw \
        -batch_size 1 \
        -doc_size 7 \
        -lr 2e-5 \
        -gradient_accumulation_steps 1 \
        -n_warmup_steps 2000 \
        -logging_step 1 \
        -eval_every 1000