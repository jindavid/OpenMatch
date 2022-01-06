import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import OpenMatch as om
from transformers import AdamW

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import is_first_worker, DistributedEvalSampler, merge_resfile, set_dist_args, optimizer_to
from contextlib import nullcontext # from contextlib import suppress as nullcontext # for python < 3.7
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
import random
import numpy as np
logger = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='tb_logs/global_cat_test')


def dev(args, model, metric, dev_loader, device):
    rst_dict = {}
    for dev_batch in dev_loader:
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], dev_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device))
            elif args.model == 'edrm':
                batch_score, _ = model(dev_batch['query_wrd_idx'].to(device), dev_batch['query_wrd_mask'].to(device),
                                       dev_batch['doc_wrd_idx'].to(device), dev_batch['doc_wrd_mask'].to(device),
                                       dev_batch['query_ent_idx'].to(device), dev_batch['query_ent_mask'].to(device),
                                       dev_batch['doc_ent_idx'].to(device), dev_batch['doc_ent_mask'].to(device),
                                       dev_batch['query_des_idx'].to(device), dev_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(dev_batch['query_idx'].to(device), dev_batch['query_mask'].to(device),
                                       dev_batch['doc_idx'].to(device), dev_batch['doc_mask'].to(device))
            if args.task == 'classification':
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    return rst_dict, batch_score

def train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device, train_sampler=None):
    best_mes = 0.0
    global_step = 0 # steps that outside epoches
    for epoch in range(args.epoch):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch) # shuffle data for distributed
            logger.warning("current gpu local_rank {}".format(args.local_rank))

        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            
            sync_context = model.no_sync if (args.local_rank != -1 and (step+1) % args.gradient_accumulation_steps != 0) else nullcontext

            if args.model == 'bert':
                ###
                if args.task == 'global' or args.task == 'global_no_att':
                    with sync_context():
                        batch_score, logits = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                ###
                elif args.task == 'global_cat':
                     with sync_context():
                        batch_score, logits = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')

            if args.task == 'ranking':
                with sync_context():
                    if args.ranking_loss == 'margin_loss':
                        batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
                    elif args.ranking_loss == 'CE_loss':
                        batch_loss = loss_fn(torch.sigmoid(batch_score_pos-batch_score_neg), torch.ones(batch_score_neg.size()).to(device))
                    elif args.ranking_loss == 'triplet_loss':
                        logit_matrix = torch.cat([batch_score_pos.reshape([-1,1]), batch_score_neg.reshape([-1,1])], dim=1)
                        lsm = F.log_softmax(input=logit_matrix,dim=1)
                        batch_loss = torch.mean(-1.0 * lsm[:, 0])
                    elif args.ranking_loss == 'LCE_loss':
                        pass
                    ###
                    elif args.ranking_loss == 'gltr_margin_loss':
                        batch_score_pos = batch_score_pos.tanh()
                        batch_score_neg = batch_score_neg.tanh()
                        # label_tensor = train_batch['label'].repeat(len(batch_score), 1)
                        # label_tensor = label_tensor.to(device)
                        score_tensor_pos = batch_score_pos.repeat(len(batch_score_pos), 1)
                        score_tensor_neg = batch_score_neg.repeat(len(batch_score_neg), 1)
                        mask = torch.ones(score_tensor_pos.size())
                        hinge_loss = loss_fn(score_tensor_pos, score_tensor_neg.t(), torch.ones(score_tensor_neg.size()).to(device))
                        batch_loss = torch.mean(hinge_loss[mask > 0])
                    ###
            ###
            elif args.task == 'global' or args.task == 'global_no_att':

                if args.global_loss == 'margin_loss':

                    # mask = train_batch['label'].reshape(-1,1) - train_batch['label'].reshape(1,-1)
                    # mask = mask.to(device)
                    batch_score = batch_score.tanh()
                    label_tensor = train_batch['label'].repeat(len(batch_score), 1)
                    label_tensor = label_tensor.to(device)
                    mask = label_tensor - label_tensor.t()
                    score_tensor = batch_score.repeat(len(batch_score), 1)
                    hinge_loss = loss_fn(score_tensor, score_tensor.t(), torch.ones(score_tensor.size()).to(device))
                    # diff_score = batch_score.reshape(-1,1).tanh() - batch_score.reshape(1,-1).tanh()
                    # hinge_loss = loss_fn(diff_score, torch.zeros(diff_score.size()).to(device), torch.ones(diff_score.size()).to(device))
                    batch_loss = torch.mean(hinge_loss[mask > 0])

                elif args.global_loss == 'CE_loss':
                    label_tensor = train_batch['label'].repeat(len(batch_score), 1)
                    label_tensor = label_tensor.to(device)
                    mask = label_tensor - label_tensor.t()
                    # print(mask)

                    score_tensor = batch_score.repeat(len(batch_score), 1)
                    diff_tensor = score_tensor - score_tensor.t()
                    # print(diff_tensor)
                    diff_score = diff_tensor[mask > 0]
                    # print(diff_score)
                    # print(torch.sigmoid(diff_score))
                    batch_loss = loss_fn(torch.sigmoid(diff_score), torch.ones(diff_score.size()).to(device))
                    # print(batch_loss)

                    # break

            elif args.task == 'global_cat':
                continue
            ###
            elif args.task == 'classification':
                with sync_context():
                    batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')

            if args.n_gpu > 1:
                batch_loss = batch_loss.mean()
            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            avg_loss += batch_loss.item()

            with sync_context():
                batch_loss.backward()

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()
                global_step += 1

                if args.logging_step > 0 and ((global_step+1) % args.logging_step == 0 or (args.test_init_log and global_step==0)):
                    # if is_first_worker():
                    if args.local_rank in [-1,0]:
                        logger.info( "training gpu {}:,  global step: {}, local step: {}, loss: {}".format(args.local_rank,global_step+1, step+1, avg_loss/args.logging_step))
                        writer.add_scalar('avg_loss',avg_loss/args.logging_step, step)
                        
                    avg_loss = 0.0

                if (global_step+1) % args.eval_every == 0 or (args.test_init_log and global_step==0):                
                    model.eval()
                    with torch.no_grad():
                        rst_dict, batch_score_dev = dev(args, model, metric, dev_loader, device)
                    model.train()

                    if args.local_rank != -1:
                        # distributed mode, save dicts and merge
                        om.utils.save_trec(args.res + "_rank_{:03}".format(args.local_rank), rst_dict)
                        dist.barrier()
                        # if is_first_worker():
                        if args.local_rank in [-1,0]:
                            merge_resfile(args.res + "_rank_*", args.res)

                    else:
                        om.utils.save_trec(args.res, rst_dict)
                        
                    # if is_first_worker():
                    if args.local_rank in [-1,0]:
                        if args.metric.split('_')[0] == 'mrr':
                            mes = metric.get_mrr(args.qrels, args.res, args.metric)
                        else:
                            mes = metric.get_metric(args.qrels, args.res, args.metric)

                        best_mes = mes if mes >= best_mes else best_mes
                        logger.info( 'save_model at step {}'.format(global_step+1))
                        if args.n_gpu > 1:
                            torch.save(model.module.state_dict(), args.save + "_step-{}".format(global_step+1))
                        else:
                            torch.save(model.state_dict(), args.save + "_step-{}".format(global_step+1))
                        logger.info( "global step: {}, messure: {}, best messure: {}".format(global_step+1, mes, best_mes))
                        writer.add_scalar('dev', mes, step)

                        # writer.add_scalar('dev_loss', mes, step)
            # dist.barrier()  


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-ranking_loss', type=str, default='margin_loss')
    parser.add_argument('-global_loss', type=str, default='margin_loss')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-reset', action='store_true', default=False)
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-dev_eval_batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=4) 
    parser.add_argument("-max_grad_norm", default=1.0,type=float,help="Max gradient norm.",)
    parser.add_argument('-eval_every', type=int, default=1000)
    parser.add_argument('-logging_step', type=int, default=100)
    parser.add_argument('-test_init_log', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=-1) # for distributed mode
    parser.add_argument("--server_ip", type=str,default="", help="For distant debugging.",)  
    parser.add_argument("--server_port", type=str, default="",help="For distant debugging.",)

    args = parser.parse_args()

    set_dist_args(args) # get local cpu/gpu device

    args.model = args.model.lower()
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        logger.info('reading training data...')
        if args.maxp:
            print('maxp')
        else:
            train_set = om.data.datasets.BertDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        logger.info('reading dev data...')
        if args.maxp:
            dev_set = om.data.datasets.BertMaxPDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            dev_set = om.data.datasets.BertDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
               task=args.task
            )
 
    if args.local_rank != -1:
        # train_sampler = DistributedSampler(train_set, args.world_size, args.local_rank)

        if args.task == 'global' or args.task == 'global_no_att':
            train_sampler = DistributedSampler(train_set)
            train_loader = om.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1,
                sampler=train_sampler
            )
        else:
            train_sampler = DistributedSampler(train_set)
            train_loader = om.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=1,
                sampler=train_sampler
            )

        if args.task == 'global' or args.task == 'global_no_att':
            dev_sampler = DistributedEvalSampler(dev_set)
            dev_loader = om.data.DataLoader(
                dataset=dev_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1,
                sampler=dev_sampler
            )
            dist.barrier()
        else:
            #dev_sampler = DistributedSampler(dev_set)
            dev_sampler = DistributedEvalSampler(dev_set)
            dev_loader = om.data.DataLoader(
                dataset=dev_set,
                batch_size=args.batch_size, # * 16 if args.dev_eval_batch_size <= 0 else args.dev_eval_batch_size,
                shuffle=True,
                num_workers=1,
                sampler=dev_sampler
            )
            dist.barrier()

    else:
        train_loader = om.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=False, ## dj
            num_workers=8
        )
        if args.task == 'global' or args.task == 'global_no_att':
            dev_loader = om.data.DataLoader(
                dataset=dev_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8
            )
            train_sampler = None
        else:
            dev_loader = om.data.DataLoader(
                dataset=dev_set,
                batch_size=args.batch_size * 16,
                shuffle=False,
                num_workers=8
            )
            train_sampler = None
            

    if args.model == 'bert' or args.model == 'roberta':
        if args.maxp:
            model = om.models.BertMaxP(
                pretrained=args.pretrain,
                max_query_len=args.max_query_len,
                max_doc_len=args.max_doc_len,
                mode=args.mode,
                task=args.task
            )
        ###
        elif args.task == 'global' or args.task == 'global_cat':
            model = om.models.BertGlobal2(
                pretrained=args.pretrain,
                mode=args.mode,
                task=args.task,
                batch_size=args.batch_size
            )
        ###
        else:
            model = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task=args.task
            )
        if args.reinfoselect:
            policy = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task='classification'
            )

    else:
        raise ValueError('model name error.')

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        if args.model == 'bert':
            st = {}
            for k in state_dict:
                if k.startswith('bert'):
                    st['_model'+k[len('bert'):]] = state_dict[k]
                elif k.startswith('classifier'):
                    st['_dense'+k[len('classifier'):]] = state_dict[k]
                else:
                    st[k] = state_dict[k]
            model.load_state_dict(st)
        else:
            model.load_state_dict(state_dict)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device

    if args.reinfoselect:
        print('reinfo')
    else:
        if args.task == 'ranking':
            if args.ranking_loss == 'margin_loss':
                loss_fn = nn.MarginRankingLoss(margin=1)
            elif args.ranking_loss == 'CE_loss':
                loss_fn = nn.BCELoss()
            elif args.ranking_loss == 'triplet_loss':
                loss_fn = nn.BCELoss() # dummpy loss for occupation
                # loss_fn = F.log_softmax(dim=1)
            elif args.ranking_loss == 'LCE_loss':
                print("LCE loss TODO")
                # nn.CrossEntropyLoss()
            elif args.ranking_loss == 'gltr_margin_loss':
                loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
            else:
                loss_fn = nn.BCELoss() # dummpy loss for occupation

        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        ###
        elif args.task == 'global' or args.task == 'global_no_att' or args.task == 'global_cat':
            if args.global_loss == 'margin_loss':
                loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
            elif args.global_loss == 'CE_loss':
                loss_fn = nn.BCELoss()
            else:
                loss_fn = nn.BCELoss() # dummpy loss for occupation
        ###
        else:
            raise ValueError('Task must be `ranking` or `classification`.')


    model.to(device)
    loss_fn.to(device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        # dev_model = nn.DataParallel(dev_model)
        loss_fn = nn.DataParallel(loss_fn)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()

    model.zero_grad()
    model.train()
    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.local_rank == -1:
        m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size)
    else:
        m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//(args.batch_size*args.world_size*args.gradient_accumulation_steps))
    if args.reinfoselect:
        p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)

    optimizer_to(m_optim,device)
    

    metric = om.metrics.Metric()

    logger.info(args)
    if args.reinfoselect:
        print('reinfo')
    else:
        train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device, train_sampler=train_sampler)

if __name__ == "__main__":
    main()