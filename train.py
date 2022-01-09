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

writer = SummaryWriter(log_dir='tb_logs/global_cat_trail2')


def dev(args, model, metric, dev_loader, device):
    rst_dict = {}
    for dev_batch in dev_loader:
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], dev_batch['retrieval_score']
        with torch.no_grad():
            batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    return rst_dict, batch_score

def inference(args, dataset, device):

    tokenizer = AutoTokenizer.from_pretrained(args.vocab)

    inf_set = om.data.datasets.BertDataset(
        dataset=dataset[0],
        tokenizer=tokenizer,
        mode='inf',
        query_max_len=args.max_query_len,
        doc_max_len=args.max_doc_len,
        max_input=args.max_input,
        doc_size=args.doc_size,
        task=args.task
    )

    inf_loader = om.data.DataLoader(
        dataset=inf_set,
        batch_size=args.doc_size,
        shuffle=True,
        num_workers=8
    )

    model = om.models.BertGlobalCat(
        pretrained=args.pretrain,
        mode=args.mode,
        task=args.task,
        inf=True
    )

    model.to(device)
    output_cls = []
    for inf_batch in inf_loader:
        with torch.no_grad():
            cls, _ = model(inf_batch['input_ids'].to(device), inf_batch['input_mask'].to(device), inf_batch['segment_ids'].to(device))
            cls.detach()
            output_cls.append(cls)

    output_cls = torch.cat(output_cls, dim=0)

    return output_cls


def train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device, train_sampler=None):
    best_mes = 0.0
    global_step = 0 # steps that outside epoches
    for epoch in range(args.epoch):

        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):

            if args.task == 'global' or args.task == 'global_no_att':

                batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
            ###
            elif args.task == 'global_cat':
                
                output_cls = inference(args, train_batch['dataset'], device)
                batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device), output_cls.to(device))
                batch_score =  batch_score[:args.doc_size]

            if args.task == 'global' or args.task == 'global_no_att':

                label_tensor = train_batch['label'].repeat(len(batch_score), 1)
                label_tensor = label_tensor.to(device)
                mask = label_tensor - label_tensor.t()

                score_tensor = batch_score.repeat(len(batch_score), 1)
                diff_tensor = score_tensor - score_tensor.t()
                diff_score = diff_tensor[mask > 0]

                batch_loss = loss_fn(torch.sigmoid(diff_score), torch.ones(diff_score.size()).to(device))

            elif args.task == 'global_cat':
                label_tensor = train_batch['label'].repeat(len(batch_score), 1)
                label_tensor = label_tensor.to(device)
                mask = label_tensor - label_tensor.t()

                score_tensor = batch_score.repeat(len(batch_score), 1)
                diff_tensor = score_tensor - score_tensor.t()
                diff_score = diff_tensor[mask > 0]
                batch_loss = loss_fn(torch.sigmoid(diff_score), torch.ones(diff_score.size()).to(device))

            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            avg_loss += batch_loss.item()

            batch_loss.backward()

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()
                global_step += 1

                if args.logging_step > 0 and ((global_step+1) % args.logging_step == 0 or (args.test_init_log and global_step==0)):
                    logger.info( "training gpu {}:,  global step: {}, local step: {}, loss: {}".format(-1,global_step+1, step+1, avg_loss/args.logging_step))
                    writer.add_scalar('avg_loss',avg_loss/args.logging_step, step)
                        
                    avg_loss = 0.0

                if (global_step+1) % args.eval_every == 0 or (args.test_init_log and global_step==0):                
                    model.eval()
                    with torch.no_grad():
                        rst_dict, _ = dev(args, model, metric, dev_loader, device)
                    model.train()

                    om.utils.save_trec(args.res, rst_dict)
                    
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-optimizer', type=str, default='adam')
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
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-doc_size', type=int, default=5)  ### no grad detach docs
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=4) 
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

    tokenizer = AutoTokenizer.from_pretrained(args.vocab)

    logger.info('reading training data...')
    train_set = om.data.datasets.BertDataset(
        dataset=args.train,
        tokenizer=tokenizer,
        mode='train',
        query_max_len=args.max_query_len,
        doc_max_len=args.max_doc_len,
        max_input=args.max_input,
        doc_size=args.doc_size,
        task=args.task
    )

    logger.info('reading dev data...')
    dev_set = om.data.datasets.BertDataset(
        dataset=args.dev,
        tokenizer=tokenizer,
        mode='dev',
        query_max_len=args.max_query_len,
        doc_max_len=args.max_doc_len,
        max_input=args.max_input,
        task=args.task
    )

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

        model = om.models.BertGlobal2(
            pretrained=args.pretrain,
            mode=args.mode,
            task=args.task,
            batch_size=args.batch_size
        )

        train_sampler = None

    elif args.task == 'global_cat':
        dev_loader = om.data.DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8
        )

        model = om.models.BertGlobalCat(
            pretrained=args.pretrain,
            mode=args.mode,
            task=args.task
        )

        train_sampler = None

    device = args.device

    loss_fn = nn.BCELoss()
    loss_fn.to(device)

    model.to(device)
    model.zero_grad()
    model.train()

    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size)

    optimizer_to(m_optim,device)

    metric = om.metrics.Metric()

    logger.info(args)
    train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device, train_sampler=train_sampler)

if __name__ == "__main__":
    main()