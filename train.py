import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='1', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('--bs',type=int,default=None)
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_memlab import LineProfiler, profile
from torchstat import stat

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# torch.cuda.memory._record_memory_history(enabled=True)

# mem_prof = LineProfiler()
# mem_prof.enable()

# set_seed(0)

node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
# g:tcsr df:feature
g, df = load_graph(args.data)

sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
if args.bs is not None:
    print(f"batch size is {args.bs}")
    train_param["batch_size"] = args.bs

def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set
    
    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end+index)
    
    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds

if args.use_inductive:
    inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]
    
gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
# TODO always false
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
# for p in model.parameters():
#     print(p, p.dtype)
#     break
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
# model_sum = sum(p.numel() for p in model.parameters())
# print(model_sum * 4 / 1024/1024)
# exit()

creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
feat_params = 0
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        feat_params += node_feats.numel()
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        feat_params += edge_feats.numel()
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()
        
print(f"feat_params: {feat_params * 4 / 1024 / 1024}")

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSamplerWrap(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.src.values)
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

def eval(mode='val'):
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

if not os.path.isdir('models'):
    os.mkdir('models')
if args.model_name == '':
    path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
else:
    path_saver = 'models/{}.pkl'.format(args.model_name)
best_ap = 0
best_e = 0
val_losses = list()
group_indexes = list()
group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))
if 'reorder' in train_param:
    # random chunk shceduling
    reorder = train_param['reorder']
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
    group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, train_param['reorder']):
        additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])

# reset timer
sampler.reset_statistic()
model.reset_time()
if mailbox is not None:
    mailbox.reset_time()

time_forward = 0
time_backward = 0
time_total = 0
time_update_mail = 0
time_update_mem = 0

time_tot_list = []
time_sample_list = []
time_prep_list = []
time_prep_mail_list = []
time_backward_list = []
time_forward_list = []
time_update_mail_list = []
time_update_mem_list = []
time_other_list = []


n_epoch = train_param['epoch']

profile_log = "all_gpu_{}".format(args.data)
need_profile = False
if need_profile:
    prof = torch.profiler.profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                                    schedule=torch.profiler.schedule(wait=3,warmup=2,active=30,repeat=1),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_log/{}'.format(profile_log)),
                                    profile_memory=True,
                                    record_shapes=True,
                                    with_stack=True)
    prof.start()

early_stop = EarlyStopMonitor(3)
# for e in range(train_param['epoch']):
for e in range(n_epoch):
    print('Epoch {:d}:'.format(e))
    time_sample = 0
    time_prep = 0
    time_prep_mail = 0
    time_tot = 0
    total_loss = 0
    time_backward = 0
    time_forward = 0
    time_update_mail = 0
    time_update_mem = 0

    # training
    model.train()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
    # batch训练
    # len(group_indexes) 好像就是1,我不知道为什么这里要这样写,所以这里就是根据batch_id每次取一个batch训练
    for i, (_, rows) in enumerate(df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)])):
        if need_profile:
            prof.step()
            if i > 3+ 2 +20:
                prof.stop()
                print(f"profile end")
                exit(0)
        t_tot_s = time.time()
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        # sample
        with record_function("sample"):
            t_sample_start = time.time()
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
                # time_sample += ret[0].sample_time()
                #time_sample += ret[0].tot_time()
            time_sample += time.time() - t_sample_start

        t_prep_s = time.time()
        # 把sample的结果变成DGLblock
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
        t_prep_mail_start = time.time()
        time_prep += (t_prep_mail_start - t_prep_s)
        # 准备src节点的memory,memory_ts,mails,mail_ts
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        time_prep_mail += (time.time() - t_prep_mail_start)

        t_forward = time.time()
        optimizer.zero_grad()
        # 前向包括update memory和embedding
        pred_pos, pred_neg = model(mfgs)
        time_forward += time.time() - t_forward
        
        t_backward = time.time()
        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * train_param['batch_size']
        # backward
        loss.backward()
        optimizer.step()
        time_backward += (time.time() - t_backward)
        
        t_update_mail_start = time.time()
        # 更新memory和mailbox
        if mailbox is not None:
            eid = rows['Unnamed: 0'].values
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
            t_update_mem_start = time.time()
            time_update_mail += t_update_mem_start -  t_update_mail_start
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
            time_update_mem += time.time() - t_update_mem_start
        time_tot += time.time() - t_tot_s
    ap, auc = eval('val')
    if e > 2 and ap > best_ap:
        best_e = e
        best_ap = ap
        torch.save(model.state_dict(), path_saver)
    time_total += time_tot
    time_other = time_tot - time_sample - time_prep - time_prep_mail - time_forward - time_backward - time_update_mem - time_update_mail
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    print('\ttotal time:{:.3f}s sample time:{:.3f}s prep time:{:.3f}s prep mail time: {:.3f}s'.format(time_tot, time_sample, time_prep, time_prep_mail))
    print('\tforward time:{:.3f}s backward time: {:.3f}s update mem time : {:.3f}s update mail time: {:.3f}s'.format(time_forward, time_backward, time_update_mem,time_update_mail))
    print('\tother time: {:.3f}s'.format(time_other))
    time_tot_list.append(time_tot)
    time_sample_list.append(time_sample)
    time_prep_list.append(time_prep)
    time_prep_mail_list.append(time_prep_mail)
    time_forward_list.append(time_forward)
    time_backward_list.append(time_backward)
    time_update_mem_list.append(time_update_mem)
    time_update_mail_list.append(time_update_mail)
    time_other_list.append(time_other)
    if early_stop.early_stop_check(ap):
        print(f"ap don't improve in 3 epoches, early stop")
        break
    

def mean(arr):
    return sum(arr[1:]) / len(arr[1:])

print(f"average statistics: ")
print('\ttotal time:{:.3f}s sample time:{:.3f}s prep time:{:.3f}s prep mail time: {:.3f}s'.format(mean(time_tot_list), mean(time_sample_list), mean(time_prep_list), mean(time_prep_mail_list)))
print('\tforward time:{:.3f}s backward time: {:.3f}s update mem time : {:.3f}s update mail time: {:.3f}s'.format(mean(time_forward_list), mean(time_backward_list), mean(time_update_mem_list),mean(time_update_mail_list)))
print('\tother time: {:.3f}s'.format(mean(time_other_list)))

# mem_prof.disable()
# mem_prof.print_stats()

# from pytorch_memlab import MemReporter

# reporter = MemReporter()
# reporter.report()

# exit()

    
print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(path_saver))
model.eval()
if sampler is not None:
    sampler.reset()
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None
    eval('train')
    eval('val')
ap, auc = eval('test')
if args.eval_neg_samples > 1:
    print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
else:
    print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
