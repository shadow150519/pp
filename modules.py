import time

import torch
import dgl
from memorys import *
from layers import *

class GeneralModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False,device_id=0):
        super(GeneralModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        self.device_id = device_id

        # input dim: 2 * memory_param['dim_out'] + dim_edge
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']): # ，每个snapshot会有一个Transformer
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined,device_id=self.device_id)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False,device_id=self.device_id)
        elif gnn_param['arch'] == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
                if 'time_transform' in gnn_param and gnn_param['time_transform'] == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param['dim_out'])
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])

        # 计算时间
        self.time_elapsed = 0.0

    def reset_time(self):
        self.time_elapsed = 0.0
        self.memory_updater.reset_time()

    def report_statistic(self,n_epoch,mailbox_time_memory):
        print(f"\t time_forward: {self.time_elapsed/n_epoch:.3f}")
        print(f"\t time_memory: {(self.memory_updater.time_memory+mailbox_time_memory)/n_epoch:.3f}")
        print(f"\t time_message: {(mailbox_time_memory)/n_epoch:.3f}")

    def forward(self, mfgs, neg_samples=1):
        import time

        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0]) # 用t-1batch的message更新memory
        t_start = time.time()
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)

        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        result = self.edge_predictor(out, neg_samples=neg_samples)
        self.time_elapsed += time.time() - t_start
        return result

    def get_emb(self, mfgs):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out

class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x