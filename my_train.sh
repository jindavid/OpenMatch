CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train.py \
        -task ranking -ranking_loss triplet_loss \
        -model bert \
        -train queries=/data2/jindawei/data/msmarco-doctrain-queries.tsv,docs=/data2/jindawei/data/msmarco-docs-firstP_title.tsv,qrels=/data2/jindawei/data/msmarco/rerank_doc/msmarco-doctrain-qrels.tsv,trec=/data2/jindawei/data/trids_marco-doc_ance-firstp-norm-dual-v6.random-20-from-top-100.zkt_code.tsv \
        -max_input 40000000 \
        -save ./checkpoints/gltr_run_no_test \
        -dev /data2/liyz/OpenMatch_gitee/data/dev_valid_1200_subset_from-splitidx-0.top100.jsonl \
        -qrels /data2/jindawei/data/msmarco-docdev-qrels.tsv \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -res ./results/gltr_run_no_test.trec \
        -metric ndcg_cut_100 \
        -max_query_len 64 \
        -max_doc_len 445 \
        -epoch 1 \
        -optimizer adamw \
        -batch_size 4 \
        -lr 2e-5 \
        -n_warmup_steps 2000 \
        -logging_step 100 \
        -eval_every 10000