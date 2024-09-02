import torch
import dgl
from layers import TimeEncode
from torch_scatter import scatter
import time
from gpucache.cache_wrapper import CacheWrapper
from typing import Dict

from pytorch_memlab import profile

class MailBox():
    def __init__(self, memory_param, num_nodes, dim_edge_feat, _node_memory=None, _node_memory_ts=None,_mailbox=None, _mailbox_ts=None, _next_mail_pos=None, _update_mail_pos=None, use_cache=False, mem_cache_size=0, mail_cache_size=0, mem_strategy="LRU",mail_strategy="LRU",device_id=0, max_query_num=0):
        self.memory_param = memory_param
        self.dim_edge_feat = dim_edge_feat
        if memory_param['type'] != 'node':
            raise NotImplementedError
        self.node_memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32) if _node_memory is None else _node_memory
        self.node_memory_ts = torch.zeros(num_nodes, dtype=torch.float32) if _node_memory_ts is None else _node_memory_ts
        self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32) if _mailbox is None else _mailbox
        self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32) if _mailbox_ts is None else _mailbox_ts
        self.next_mail_pos = torch.zeros((num_nodes), dtype=torch.long) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
        self.device = torch.device('cpu')
        self.time_message = 0
        self.time_memory = 0
        self.mailbox_cache = {}
        self.device_id = device_id
        self.use_cache = use_cache
        if use_cache:
            print("mailbox use cache")
            print(f"mail box max query: {max_query_num}")
            self.mailbox_cache['mem'] = CacheWrapper(self.node_memory,mem_cache_size,mem_strategy,memory_param['dim_out'],max_query_num,device_id,name="memory cache")
            self.mailbox_cache["mem_ts"] = CacheWrapper(self.node_memory_ts,mem_cache_size,mem_strategy,1,max_query_num,device_id,name="memory_ts cache")
            self.mailbox_cache['mem_input'] = CacheWrapper(self.mailbox.reshape(num_nodes,-1), mail_cache_size,mail_strategy,memory_param['mailbox_size'] * (2 * memory_param['dim_out'] + dim_edge_feat),max_query_num,device_id,name="mail cache")
            self.mailbox_cache['mail_ts'] = CacheWrapper(self.mailbox_ts.reshape(num_nodes,-1), mail_cache_size,mail_strategy,memory_param['mailbox_size'] ,max_query_num,device_id,name="mail_ts cache")

    def report_cache_usage(self):
        for k, cache in self.mailbox_cache.items():
            cache.report_time()
    def reset_cache(self):
        for k in self.mailbox_cache.keys():
            self.mailbox_cache[k].reset_cache()

    def reset_time(self):
        self.time_message = 0
        self.time_memory = 0
        
    def reset(self):
        self.node_memory.fill_(0)
        self.node_memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)
        
    
    def move_to_gpu(self, device_id=0):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mailbox = self.mailbox.cuda()
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        sum_param = (self.node_memory.numel() + self.node_memory_ts.numel() + self.mailbox.numel() + self.mailbox_ts.numel() + self.next_mail_pos.numel()) * 4 / 1024 / 1024
        print(f"mailbox params: {sum_param}")
        self.device = torch.device('cuda', device_id)

    def allocate_pinned_memory_buffers(self, sample_param, batch_size):
        limit = int(batch_size * 3.3)
        if 'neighbor' in sample_param:
            for i in sample_param['neighbor']:
                limit *= i + 1
        self.pinned_node_memory_buffs = list()
        self.pinned_node_memory_ts_buffs = list()
        self.pinned_mailbox_buffs = list()
        self.pinned_mailbox_ts_buffs = list()
        for _ in range(sample_param['history']):
            self.pinned_node_memory_buffs.append(torch.zeros((limit, self.node_memory.shape[1]), pin_memory=True))
            self.pinned_node_memory_ts_buffs.append(torch.zeros((limit,), pin_memory=True))
            self.pinned_mailbox_buffs.append(torch.zeros((limit, self.mailbox.shape[1], self.mailbox.shape[2]), pin_memory=True))
            self.pinned_mailbox_ts_buffs.append(torch.zeros((limit, self.mailbox_ts.shape[1]), pin_memory=True))

    
    def prep_input_mails(self, mfg, use_pinned_buffers=False, device_id=0):
        # t_start = time.time()
        for i, b in enumerate(mfg):
            if use_pinned_buffers:
                idx = b.srcdata['ID'].cpu().long()
                torch.index_select(self.node_memory, 0, idx, out=self.pinned_node_memory_buffs[i][:idx.shape[0]])
                b.srcdata['mem'] = self.pinned_node_memory_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                torch.index_select(self.node_memory_ts,0, idx, out=self.pinned_node_memory_ts_buffs[i][:idx.shape[0]])
                b.srcdata['mem_ts'] = self.pinned_node_memory_ts_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                torch.index_select(self.mailbox, 0, idx, out=self.pinned_mailbox_buffs[i][:idx.shape[0]])
                b.srcdata['mem_input'] = self.pinned_mailbox_buffs[i][:idx.shape[0]].reshape(b.srcdata['ID'].shape[0], -1).cuda(non_blocking=True)
                torch.index_select(self.mailbox_ts, 0, idx, out=self.pinned_mailbox_ts_buffs[i][:idx.shape[0]])
                b.srcdata['mail_ts'] = self.pinned_mailbox_ts_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
            elif self.use_cache:
                idx = b.srcdata['ID']
                b.srcdata['mem'] = self.mailbox_cache["mem"].Get(idx)
                b.srcdata['mem_ts'] = self.mailbox_cache["mem_ts"].Get(idx).reshape(-1)
                b.srcdata['mem_input'] = self.mailbox_cache["mem_input"].Get(idx)
                b.srcdata['mail_ts'] = self.mailbox_cache["mail_ts"].Get(idx)
            else:
                # 准备好src data的memory, ts, mails, mail_ts
                idx = b.srcdata['ID'].cpu()
                idx = idx.long()
                b.srcdata['mem'] = self.node_memory[idx].cuda(device_id)
                b.srcdata['mem_ts'] = self.node_memory_ts[idx].cuda(device_id)
                b.srcdata['mem_input'] = self.mailbox[idx].cuda(device_id).reshape(b.srcdata['ID'].shape[0], -1)
                b.srcdata['mail_ts'] = self.mailbox_ts[idx].cuda(device_id)
            # print(f"b.srcdata['mem'].shape {b.srcdata['mem'].shape}")
            # print(f"b.srcdata['mem_ts'].shape {b.srcdata['mem_ts'].shape}")
            # print(f"b.srcdata['mem_input'].shape {b.srcdata['mem_input'].shape}")
            # print(f"b.srcdata['mail_ts'].shape {b.srcdata['mail_ts'].shape}")
            # print("end")
            # exit()
        # self.time_message += time.time() - t_start

    
    def update_memory(self, nid, memory, root_nodes, ts, neg_samples=1):
        t_start = time.time()
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst].to(self.device)
            memory = memory[:num_true_src_dst].to(self.device)
            ts = ts[:num_true_src_dst].to(self.device)
            self.node_memory[nid.long()] = memory
            self.node_memory_ts[nid.long()] = ts
        time_elapsed = time.time() - t_start
        self.time_memory += time_elapsed
        # print(f"writeback memory elapsed: {time_elapsed * 1000:.3f} ms")
    
    def update_mailbox(self, nid, memory, root_nodes, ts, edge_feats, block, neg_samples=1):
        # t_start = time.time()
        with torch.no_grad():
            num_true_edges = root_nodes.shape[0] // (neg_samples + 2)
            memory = memory.to(self.device)
            if edge_feats is not None:
                edge_feats = edge_feats.to(self.device)
            if block is not None:
                block = block.to(self.device)
            # TGN/JODIE
            if self.memory_param['deliver_to'] == 'self':
                src = torch.from_numpy(root_nodes[:num_true_edges]).to(self.device)
                dst = torch.from_numpy(root_nodes[num_true_edges:num_true_edges * 2]).to(self.device)
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1]) # mail shape: 2 shape (2 * len(src_mail), src_mail.shape[1])
                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                mail_ts = torch.from_numpy(ts[:num_true_edges * 2]).to(self.device)
                if mail_ts.dtype == torch.float64:
                    import pdb; pdb.set_trace()
                # find unique nid to update mailbox
                uni, inv = torch.unique(nid, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]
                if self.memory_param['mail_combine'] == 'last':
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                    if self.memory_param['mailbox_size'] > 1:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
            # APAN
            elif self.memory_param['deliver_to'] == 'neighbors':
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=0)
                mail = torch.cat([mail, mail[block.edges()[0].long()]], dim=0)
                mail_ts = torch.from_numpy(ts[:num_true_edges * 2]).to(self.device)
                mail_ts = torch.cat([mail_ts, mail_ts[block.edges()[0].long()]], dim=0)
                if self.memory_param['mail_combine'] == 'mean':
                    (nid, idx) = torch.unique(block.dstdata['ID'], return_inverse=True)
                    mail = scatter(mail, idx, reduce='mean', dim=0)
                    mail_ts = scatter(mail_ts, idx, reduce='mean')
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                elif self.memory_param['mail_combine'] == 'last':
                    nid = block.dstdata['ID']
                    # find unique nid to update mailbox
                    uni, inv = torch.unique(nid, return_inverse=True)
                    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                    perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                    nid = nid[perm]
                    mail = mail[perm]
                    mail_ts = mail_ts[perm]
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                else:
                    raise NotImplementedError
                if self.memory_param['mailbox_size'] > 1:
                    if self.update_mail_pos is None:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
                    else:
                        self.update_mail_pos[nid.long()] = 1
            else:
                raise NotImplementedError
        # time_elapsed = time.time() - t_start
        # self.time_message += time_elapsed
        #print(f"compute message elapsed: {time_elapsed * 1000:.3f} ms")
    def update_next_mail_pos(self):
        # t_start = time.time()
        if self.update_mail_pos is not None:
            nid = torch.where(self.update_mail_pos == 1)[0]
            self.next_mail_pos[nid] = torch.remainder(self.next_mail_pos[nid] + 1, self.memory_param['mailbox_size'])
            self.update_mail_pos.fill_(0)
        # time_elapsed = time.time() - t_start
        # self.time_message += time_elapsed
        #print(f"update next message pos message elapsed: {time_elapsed * 1000:.3f} ms")

class GRUMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

        self.time_memory = 0

    def reset_time(self):
        self.time_memory = 0

    def forward(self, mfg):
        t_start = time.time()
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            # print(f"memory compute message size: {b.srcdata['mem_input'].shape}, memory_size: {b.srcdata['mem'].shape}")
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory
        time_elapsed = time.time() - t_start
        # print(f"update memory elapsed: {time_elapsed * 1000:.3f} ms")
        self.time_memory += time_elapsed

class RNNMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(RNNMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.RNNCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
        self.time_memory = 0

    def reset_time(self):
         self.time_memory = 0


    def forward(self, mfg):
        t_start = time.time()
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            # print(f"memory compute message size: {b.srcdata['mem_input'].shape}, memory_size: {b.srcdata['mem'].shape}")
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory
        self.time_memory += time.time() - t_start

class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_out, dim_time, train_param):
        super(TransformerMemoryUpdater, self).__init__()
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.att_h = memory_param['attention_head']
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(train_param['dropout'])
        self.att_dropout = torch.nn.Dropout(train_param['att_dropout'])
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        self.time_memory = 0

    def reset_time(self):
        self.time_memory = 0

    def forward(self, mfg):
        t_start = time.time()
        for b in mfg:
            Q = self.w_q(b.srcdata['mem']).reshape((b.num_src_nodes(), self.att_h, -1))
            mails = b.srcdata['mem_input'].reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.srcdata['mail_ts']).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            V = self.w_v(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:,:,:,None]*V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += b.srcdata['mem']
            rst = self.layer_norm(rst)
            rst = self.mlp(rst)
            rst = self.dropout(rst)
            rst = torch.nn.functional.relu(rst)
            b.srcdata['h'] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
        self.time_memory += time.time() - t_start
