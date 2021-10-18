CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch \
--nproc_per_node=4 --master_port=42718 train.py \
        -task ranking -ranking_loss triplet_loss \
        -model bert \
        -train queries=/data2/jindawei/data/msmarco-doctrain-queries.tsv,docs=/data2/jindawei/data/msmarco/rerank_doc/msmarco-docs-firstP_title.tsv,qrels=/data2/jindawei/data/msmarco/rerank_doc/msmarco-doctrain-qrels.tsv,trec=/data2/jindawei/data/trids_marco-doc_ance-firstp-norm-dual-v6.random-20-from-top-100.zkt_code_global.tsv \
        -max_input 40000000 \
        -save ./checkpoints/bert-base_marco-doc_ance-firstp-norm-dual-v6.random-20-from-top-100.v24 \
        -dev /home3/liyz/OpenMatch_gitee/data/dev_valid_1200_subset_from-splitidx-0.top100.jsonl \
        -qrels /home3/liyz/ANCE_long/data/raw_data/msmarco-docdev-qrels.tsv \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -res ./results/reproduce_zkt.marco-doc_ance-firstp-norm-dual-v6.v25.trec \
        -metric ndcg_cut_100 \
        -max_query_len 64 \
        -max_doc_len 445 \
        -epoch 1 \
        -optimizer adamw \
        -batch_size 4 \
        -dev_eval_batch_size 128 \
        -gradient_accumulation_steps 4 \
        -lr 2e-5 \
        -n_warmup_steps 2000 \
        -logging_step 100 \
        -eval_every 10000

# CUDA_VISIBLE_DEVICES=5,6,7 \
# python -u -m torch.disuted.launch --nproc_per_node=3 --master_port=42787 train.py \
#         -task ranking -ranking_loss triplet_loss \
#         -model bert \
#         -train queries=/home1/public/datasets/msmarco/rerank_doc/msmarco-doctrain-queries.tsv,docs=/home1/public/datasets/msmarco/rerank_doc/msmarco-docs-firstP_title.tsv,qrels=/home1/public/datasets/msmarco/rerank_doc/msmarco-doctrain-qrels.tsv,trec=/home3/liyz/OpenMatch_gitee/data/trids_marco-doc_ance-firstp-norm-dual-v6.random-20-from-top-100.zkt_code.tsv \
#         -max_input 40000000 \
#         -save ./checkpoints/bert-base_marco-doc_ance-firstp-norm-dual-v6.random-20-from-top-100.v24 \
#         -dev  /home3/liyz/OpenMatch_gitee/data/dev_valid_1200_subset_from-splitidx-0.top100.jsonl \
#         -qrels /home3/liyz/ANCE_long/data/raw_data/msmarco-docdev-qrels.tsv \
#         -vocab bert-base-uncased \
#         -pretrain bert-base-uncased \
#         -res ./results/reproduce_zkt.marco-doc_ance-firstp-norm-dual-v6.v25.trec \
#         -metric ndcg_cut_100 \
#         -max_query_len 64 \
#         -max_doc_len 445 \
#         -epoch 1 \
#         -optimizer adamw \
#         -batch_size 4 \
#         -dev_eval_batch_size 128 \
#         -gradient_accumulation_steps 4 \
#         -lr 2e-5 \
#         -n_warmup_steps 2000 \
#         -logging_step 100 \
#         -eval_every 10 > logs/distributed_train.v24.firstp.msmarco_doc.triplet_loss.random-20-from-top-100.epoch_1.world_size_4.bsz_4.grad_accd_4.lr2e-5.warmup_2000.log1 