import argparse
import os
import pickle

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('--ncsize',type=int)
parser.add_argument('--ncstrategy',type=str)
parser.add_argument('--ecsize',type=int)
parser.add_argument('--ecstrategy',type=str)
parser.add_argument("--mailsize", type=int)
parser.add_argument("--memsize",type=int)
parser.add_argument('--disn',action="store_true")
parser.add_argument('--dise',action="store_true")
parser.add_argument('--dism',action="store_true")
args=parser.parse_args()

disable_edge = args.dise
disable_node = args.disn
disable_mail = args.dism
print(f"disable_node: {disable_node}")
print(f"disable_edge: {disable_edge}")

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from gpucache.cache_wrapper import CacheWrapper

early_stop = EarlyStopMonitor(3)
# set_seed(0)
node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
node_cache, edge_cache = None, None
node_cache_size, node_cache_strategy = args.ncsize, args.ncstrategy
edge_cache_size, edge_cache_strategy = 2 ** args.ecsize, args.ecstrategy
mail_cache_size, mem_cache_size = args.mailsize, args.memsize
    

# g:tcsr df:feature
g, df = load_graph(args.data)
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
# g = renumber_nodes_and_edges(g)
device_id = int(args.gpu)
print(f"device_id is {device_id}")
if not disable_node and node_feats is not None:
    print("use node feature")
    node_cache = CacheWrapper(node_feats, node_cache_size, node_cache_strategy,node_feats.shape[1],node_cache_size,device_id,name="node cache")
if not disable_edge and edge_feats is not None:
    print("use edge feature")
    edge_cache = CacheWrapper(edge_feats, edge_cache_size, edge_cache_strategy, edge_feats.shape[1], edge_cache_size * 4, device_id,name="edge cache")



train_edge_end = df[df['ext_roll'].gt(0)].index[0]

val_edge_end = df[df['ext_roll'].gt(1)].index[0]

print(f"Graph Dataset {args.data}:")
print(f"num_nodes {g['indptr'].shape[0] - 1}, num_edge {len(df)}")
print(f"mem size {memory_param['dim_out']}, mail size {memory_param['mailbox_size'] * (2 * memory_param['dim_out'] + edge_feats.shape[1])}, each node cache {memory_param['mailbox_size']} mails")
print(f"total {train_edge_end} edges for train, can cache {edge_cache_size}, {edge_cache_size/train_edge_end*100}%")
print(f"total {g['indptr'].shape[0] - 1} nodes for train, can cache {node_cache_size}, {node_cache_size/(g['indptr'].shape[0] - 1)*100}%")



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
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first, device_id=device_id).cuda(device_id)
# print(model)
mem_cache_size = mail_cache_size = node_cache_size
max_query_num_mailbox = train_param["batch_size"] * 3 * (1 + 10)
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge, use_cache=False, mem_cache_size=mem_cache_size,mail_cache_size=mail_cache_size,mem_strategy=node_cache_strategy,mail_strategy=node_cache_strategy,device_id=device_id,max_query_num=max_query_num_mailbox) if memory_param['type'] != 'none' else None
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

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
                mfgs = to_dgl_blocks(ret, sample_param['history'],device_id=device_id)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts,device_id=device_id)
            # print(f"disable_node: {disable_node}")
            mfgs = prepare_input2(mfgs, node_feats, edge_feats, combine_first=combine_first,nfeat_buff=node_cache,efeat_buff=edge_cache,disable_edge=disable_edge,disable_node=disable_node,device_id=device_id)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0],device_id=device_id)
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
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True,device_id=device_id)[0][0]
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
mailbox.reset_time()

time_forward = 0
time_backward = 0
time_total = 0
time_prep = 0

time_prep_list = []
need_prof = True
n_epoch = train_param['epoch']
profile_log_dir='{}_{}_{}'.format(args.data,args.ecstrategy,args.ecsize)
prof = None
if need_prof:
    prof = torch.profiler.profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                                  schedule=torch.profiler.schedule(wait=3,warmup=2,active=20,repeat=1),
                                  on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_log/new_{}_with_edge_and_node'.format(profile_log_dir)),
                                  profile_memory=True,
                                  record_shapes=True,
                                  with_stack=True)
    prof.start()
for e in range(train_param['epoch']):
    if edge_cache is not None:
        edge_cache.reset_cache()
    if node_cache is not None:
        node_cache.reset_cache()
    if mailbox.use_cache:
        mailbox.reset_cache()    
    print('Epoch {:d}:'.format(e))
    time_sample = 0
    time_prep = 0
    time_tot = 0
    total_loss = 0
    time_prep_input2 = 0
    time_prep_mail = 0
    time_build_dgl_block = 0

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
        if need_prof:
            prof.step()
            if i > (3 + 2 + 20):
                print("profile end")
                prof.stop()
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                exit()
        t_tot_s = time.time()
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        # sample
        with record_function("sample"):
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']: 
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
                # time_sample += ret[0].sample_time()
                time_sample += ret[0].tot_time()
        t_prep_s = time.time()
        # 把sample的结果变成DGLblock
        with record_function("to_dgl_block"):
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'],device_id=device_id)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts,device_id=device_id)
        #time_build_dgl_block += time.time() - t_prep_s

        #t_prep_input2_s = time.time()
        with record_function("preppare_input2"): 
            mfgs = prepare_input2(mfgs, node_feats, edge_feats, combine_first=combine_first,nfeat_buff=node_cache,efeat_buff=edge_cache,disable_edge=disable_edge,disable_node=disable_node,device_id=device_id)
        # 准备src节点的memory,memory_ts,mails,mail_ts
        #time_prep_input2 += time.time() - t_prep_input2_s 

         #t_prep_mail_s = time.time() 
        with record_function("prep_mail"):     
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0],device_id=device_id)
        #time_prep_mail += time.time() - t_prep_mail_s

        time_prep += time.time() - t_prep_s
        
        optimizer.zero_grad()

        # 前向包括update memory和embedding
        with record_function("forward"):
            t_forward = time.time()
            pred_pos, pred_neg = model(mfgs)
            time_forward += time.time() - t_forward

        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * train_param['batch_size']
        # backward
        with record_function("backward"):
            t_backward = time.time()
            loss.backward()
            optimizer.step()
            time_backward += time.time() - t_backward
        t_prep_s = time.time()
        # 更新memory和mailbox
        with record_function("update_mem_and_mailbox"):
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = None
                if edge_cache is not None:
                    mem_edge_feats = edge_cache.Get(eid)
                else:
                    mem_edge_feats = edge_feats[eid] if edge_feats is not None else None

                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True,device_id=device_id)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
        time_prep += time.time() - t_prep_s
        time_tot += time.time() - t_tot_s
    ap, auc = eval('val')
    if ap > best_ap:
        best_e = e
        best_ap = ap
        torch.save(model.state_dict(), path_saver)
        print(f"save model at epoch {e}")
    time_total += time_tot
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    print('\ttotal time:{:.3f}s sample time:{:.3f}s prep time:{:.3f}s'.format(time_tot, time_sample, time_prep))
    print('\tbuild dgl blocks time:{:.3f}s prepare input2 time {:.3f}s prepare mails time {:.3f}'.format(time_build_dgl_block,time_prep_input2,time_prep_mail))
    if edge_cache is not None:
        edge_cache.report_time()
    if node_cache is not None:
        node_cache.report_time()
    if mailbox.use_cache:
        mailbox.report_cache_usage()
    time_prep_list.append(time_prep)
    if early_stop.early_stop_check(ap):
        print(f"ap don't improve in 3 epoches, early stop")
        break
 


print(f"average statistics: ")
print(f"\t average_prep time: {sum(time_prep_list)/len(time_prep_list)}")
print(f"\t time_total {time_total/n_epoch:.3f} s")
sampler.report_statistic(n_epoch)
model.report_statistic(n_epoch, mailbox.time_memory)
print(f"\t time_backward {time_backward/n_epoch:.3f}s")
print(f"\t time_message {mailbox.time_message/n_epoch:.3f}s")

node_cache_args = {"size": 12,"strategy":"LRU"}
edge_cache_args = {"size": 12412, "strategy":"LRU"}
dataset_args = {"name":"WIKI"}
file_name = "results/{}_{}.pkl".format(args.data,edge_cache_args["size"],edge_cache_args["strategy"])
with open(file_name,"wb") as f:
    pickle.dump([node_cache_args,edge_cache_args,dataset_args,time_prep_list],f)

    
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
