import pymetis
import numpy as np
from typing import Callable, Optional, Tuple, Union, Dict, List, Iterable
from collections import defaultdict
import torch
from utils import load_graph
from torch.utils.data import Dataset, DataLoader
from collections import deque
from dataset import *
import dgl
from dgl import DGLGraph
import matplotlib.pyplot as plt
import networkx as nx
from partition2 import TCSR, Partition, TEdge
from typing import Tuple, List, Set
import math



# class Partition:
#     def __init__(self, num_nodes, edges, eids, ets, node_map):
#         self.num_nodes = num_nodes
#         self.num_edges = len(edges)
#         self.edges = edges  # coo format
#         self.eids = eids  # global edge id
#         self.ets = ets
#         self.node_map = node_map

#     def __len__(self):
#         return self.num_edges

#     def __getitem__(self, idx):
#         return self.node_map[self.edges[idx][0]], self.node_map[self.edges[idx][1]], self.eids[idx], self.ets[idx]


class PartitionedGraph():
    def __init__(self, graph, partition_num):
        self._partition_num = partition_num
        self._origin_graph = graph  # tcsr
        self._node_num = len(graph.ind) - 1
        self._edge_num = len(graph.nbr)
        self._n_cut = 0
        self._metis_graph = None
        # self._cluster_graph = None
        self._partitions: List[Partition] = [Partition(i,partition_num) for i in range(partition_num)]
        self._partition_array = []
        # self._node_map = {}
        # self._edge_map = {}
        self._finished = np.zeros(partition_num, dtype=np.int32)
        self._relation_matrix = np.zeros((partition_num, partition_num), dtype=np.float32)
        # self._partition_edge_matrix = []

    @property
    def partitions(self):
        return self._partitions

    @property
    def partition_array(self):
        return self._partition_array

    def num_edge(self):
        return self._edge_num

    def _compute_relation_matrix(self):
        for i in range(self._partition_num):
            # print(i)
            for j in range(self._partition_num):
                if i == j:
                    self._relation_matrix[i][j] = self._partitions[i].edge_num()
                else:
                    self._relation_matrix[i][j] = self._partitions[i].cut_edge_num(j)
                # self._relation_matrix[i][i] = 0

        # self._relation_matrix = self._relation_matrix / np.maximum(np.sum(self._relation_matrix, axis=1, keepdims=True),1)

        #     self._relation_matrix[i] = self._relation_matrix[i] / self._relation_matrix[i][i]
        #     self._relation_matrix[i][i] = -1e6
        # self._relation_matrix = np.exp(self._relation_matrix)
        # self._relation_matrix = self._relation_matrix / np.sum(self._relation_matrix, axis=1, keepdims=True)
        print(f"\nrelation matrix:")
        for row in self._relation_matrix:
            for elem in row:
                print(f"{elem:4.0f}", end=" ")
            print()
        print()

    def _to_metis_graph(self, graph) -> DGLGraph:

        indptr2, indices2, eid2 = graph.ind, graph.nbr, graph.eid
        g2 = dgl.graph(('csr', (indptr2, indices2, eid2)))
        g = dgl.to_simple(g2, return_counts="weight")
        indptr, indices, eid = g.adj_tensors('csr')
        weight = g.edata["weight"]

        # sanity check
        # for i in range(len(indptr) - 1):
        #     for j in range(indptr[i],indptr[i+1]):
        #         nbrs = indices2[indptr2[i]:indptr2[i+1]]
        #         parallel_edge_num = np.equal(nbrs, indices[j]).sum()
        #         assert parallel_edge_num == weight[j], "parallel_edge_num: {} != weight[j]: {} at index {}".format(parallel_edge_num,weight[j],j)

        # 创建一个默认字典来存储无向图的边和权重
        undirected_graph = defaultdict(int)

        # 遍历有向图的边,并创建
        for row in range(len(indptr) - 1):
            for j in range(indptr[row], indptr[row + 1]):
                col = indices[j].item()
                wgt = weight[j].item()
                undirected_graph[(row, col)] += wgt
                undirected_graph[(col, row)] += wgt

        edges = undirected_graph.keys()
        wgts = [undirected_graph[edge] for edge in edges]
        src = [edge[0] for edge in edges]
        dst = [edge[1] for edge in edges]
        g = dgl.graph((src, dst))
        g.edata["weight"] = torch.tensor(wgts, dtype=torch.int)

        return g
    

    def _partition_graph(self,alg="metis"):
        indptr, indices, _ = self._metis_graph.adj_tensors("csr")
        indptr, indices = indptr.tolist(), indices.tolist()
        rows = []
        for i in range(len(indptr) - 1):
            rows.extend([i] * (indptr[i + 1] - indptr[i]))
        ids = self._metis_graph.edge_ids(rows, indices)
        eweights = self._metis_graph.edata["weight"][ids]
        # eweights = self._metis_graph.edata["weight"].tolist()
        import time
        if alg == "metis":
            t_start = time.time()
            n_cut, partition_array = pymetis.part_graph(self._partition_num, xadj=indptr, adjncy=indices, eweights=eweights)
            t_elapsed = time.time() - t_start
            print(f"metis partition time {t_elapsed}s")
            partition_array = np.array(partition_array)
            self._n_cut, self._partition_array = n_cut, partition_array
        # elif alg == "kl":
        #     partition_array = self._kl_partition(self._networkx_graph,self._partition_num)
        # elif alg == "edge_betweenness":  
        #     partition_array = self._edge_betweenness_partition(self._networkx_graph,self._partition_num)
        # elif alg == "modularity":
        #     partition_array = self._greedy_modularity_communities(self._networkx_graph,self._partition_num)
        # elif alg == "tree":
        #     raise NotImplementedError
        #     # partition_array =
        # elif alg == "lp":
        #     partition_array = self._label_propagation_communities(self._networkx_graph)
        # elif alg == "louvain":
        #     partition_array = self._louvain_communities(self._networkx_graph)
        # elif alg == "fluidc":
        #    partition_array = self._asyn_fluidc(self._networkx_graph, self._partition_num)
        else:
            raise NotImplementedError
        # print(f"edge cut: {self._n_cut}")
        for i in range(self._partition_num):
            print(f"partition {i} has {np.count_nonzero(partition_array == i)} nodes")

        # print(f"partition array {self._partition_array}")
        # src, dst = self._origin_graph
        # removed_src = []
        # removed_dst = []
        partition_node_set = [set() for _ in range(self._partition_num)]  # (partition_num,)
        # global_to_local = [{}] * self._partition_num
        # partition_edge_matrix = [[[] for _ in range(self._partition_num)] for _ in
        #                          range(self._partition_num)]  # (partition_num,partition_num)
        
        indptr, indices, eid, ets = self._origin_graph.ind, self._origin_graph.nbr, self._origin_graph.eid, self._origin_graph.ets
        # partition edge
        for i in range(len(indptr) - 1):
            for j in range(indptr[i], indptr[i + 1]):
                src_partition, dst_partition = partition_array[i], partition_array[indices[j]]
                assert src_partition < self._partition_num, "src_partition >= self._partition_num"
                assert dst_partition < self._partition_num, "dst_partition >= self._partition_num"
                if src_partition == dst_partition:
                    self._partitions[src_partition].add_node(indices[j])
                    self._partitions[src_partition].add_node(i)
                    self._partitions[src_partition].add_edge(TEdge(i,indices[j],ets[j],eid[j]))
                else:
                    self._partitions[src_partition].add_node(i)
                    self._partitions[dst_partition].add_node(indices[j])
                    self._partitions[src_partition].add_cut_edge(TEdge(i, indices[j], ets[j], eid[j]),dst_partition)



        # remap node id
        # for i in range(self._partition_num):
        #     node_cnt = 0
        #     global_id_to_local = {}
        #     local_edges = []
        #     eids = []
        #     etss = []
        #     for u, v, eid, ets in sorted(partition_edge_matrix[i][i], key=lambda e: e[3]):
        #         if u not in global_id_to_local:
        #             global_id_to_local[u] = node_cnt
        #             node_cnt += 1
        #         if v not in global_id_to_local:
        #             global_id_to_local[v] = node_cnt
        #             node_cnt += 1
        #         local_edges.append((global_id_to_local[u], global_id_to_local[v]))
        #         eids.append(eid)
        #         etss.append(ets)
        #     assert node_cnt == len(partition_node_set[i]), "node_cnt != len(partition_node_set[i])"
        #     node_map = [0] * node_cnt
        #     for k, v in global_id_to_local.items():
        #         node_map[v] = k

        reserved_edges_num = sum([p.edge_num() for p in self._partitions])
        self._n_cut = self._edge_num - reserved_edges_num
        # print(
        #     f"total edges: {self._edge_num}, reserved_edges_num:{reserved_edges_num}, cut_edge: {self._edge_num - reserved_edges_num}({(reserved_edges_num / self._edge_num):.2f})")

        # for i in range(len(src)):
        #     if partition_array[src[i]] != partition_array[dst[i]]:
        #         removed_src.append(src[i])
        #         removed_dst.append(dst[i])
        #         # self._relation_matrix[partition_array[src[i]]]
        #     else:
        #         partition_id = partition_array[src[i]]
        #         partition_node_set[partition_id].add(src[i])
        #         partition_node_set[partition_id].add(dst[i])
        #         # if src[i] not in partition_node_set[partition_id]:
        #         #     # local_id = len(partition_node_set[partition_id])
        #         #     # global_to_local[partition_id][src[i]] = local_id
        #         #     locals[partition_id].add(src[i])
        #
        #         # if dst[i] not in locals[partition_id]:
        #         #     # local_id = len(locals[partition_id])
        #         #     # global_to_local[partition_id][dst[i]] = local_id
        #         #     locals[partition_id].add(dst[i])
        #
        #         partition_edge_lis
        # eids = g.edge_ids(removed_src,removed_dst)
        # g.remove_edges(eids)

    def prepare_graph(self,alg="metis"):
        self._metis_graph = self._to_metis_graph(self._origin_graph)
        self._networkx_graph = dgl.to_networkx(self._metis_graph,edge_attrs=['weight'])
        self._partition_graph(alg)
        self._compute_relation_matrix()
        
    def merge_partitions(self, partition_ids, add_cross_partition_edge=False):

        num_nodes = 0
        edges = []
        eids = []
        # for pid in partition_ids:

    def create_train_plan(self, parallel_partition_num=1, threshold=None, add_cross_partition_edge=False):
        scheduler = Scheduler(self._partition_num, self._relation_matrix, parallel_partition_num, threshold, self._edge_num)
        train_plan = scheduler.generate_plan()
        return train_plan
    
    # def _kl_partition(self, ng: nx.DiGraph, num: int) -> np.ndarray:
    #     # TODO: 目前的kl划分在处理最后一次划分的时候，是从上一次的分区中取第一个分区划分，可能需要做改进，比如选择边或者点最多的分区进行划分
    #     ng = ng.to_undirected()
    #     partition_queue = deque()
    #     partition_queue.append(ng)
    #     while len(partition_queue) < num:
    #         g = partition_queue.popleft()
    #         g1, g2 = nx.community.kernighan_lin_bisection(g,weight='weight')
    #         g1 = ng.subgraph(g1)
    #         g2 = ng.subgraph(g2)
    #         partition_queue.append(g1)
    #         partition_queue.append(g2)
    #
    #     partition_array = np.full(ng.number_of_nodes(),-1,dtype=np.int32)
    #     for pid, g in enumerate(partition_queue):
    #         partition_array[list(g.nodes)] = pid
    #
    #     assert np.all((partition_array >= 0) & (partition_array < self._partition_num))
    #     return partition_array
    #
    # def _edge_betweenness_partition(self,ng: nx.DiGraph, num: int) -> np.ndarray:
    #     ng = ng.to_undirected()
    #     weights = nx.get_edge_attributes(ng,"weight")
    #     weights = {k:1/v for k, v in weights.items()}
    #     nx.set_edge_attributes(ng,weights,"weight")
    #     # print(num)
    #     partitions = nx.community.edge_betweenness_partition(ng, num)
    #     # print(len(partitions))
    #     # print(partitions)
    #
    #     partition_array = np.full(ng.number_of_nodes(),-1,dtype=np.int32)
    #     for pid, p in enumerate(partitions):
    #         partition_array[list(p)] = pid
    #
    #     assert np.all((partition_array >= 0) & (partition_array < self._partition_num))
    #     return partition_array
    #
    # def _greedy_modularity_communities(self,ng: nx.DiGraph, num: int) -> np.ndarray:
    #     ng = ng.to_undirected()
    #     partition_list = nx.community.greedy_modularity_communities(ng,"weight",cutoff=num,best_n=num)
    #     partition_array = np.full(ng.number_of_nodes(),-1,dtype=np.int32)
    #     for pid, p in enumerate(partition_list):
    #         partition_array[list(p)] = pid
    #
    #     assert np.all((partition_array >= 0) & (partition_array < self._partition_num))
    #     return partition_array
    #
    # def _label_propagation_communities(self, ng: nx.DiGraph) -> np.ndarray:
    #     ng = ng.to_undirected()
    #     partition_dict = nx.community.label_propagation_communities(ng)
    #     partition_array = np.full(ng.number_of_nodes(), -1, dtype=np.int32)
    #     print(f"lp get {len(partition_dict)} communities")
    #     for pid, nodes in enumerate(partition_dict):
    #         partition_array[list(nodes)] = pid
    #     assert np.all((partition_array >= 0) & (partition_array < self._partition_num))
    #     return partition_array
    #
    # def _louvain_communities(self, ng: nx.DiGraph) -> np.ndarray:
    #     ng = ng.to_undirected()
    #     partition_dict = nx.community.louvain_communities(ng)
    #     partition_array = np.full(ng.number_of_nodes(), -1, dtype=np.int32)
    #     print(f"louvain get {len(partition_dict)} communities")
    #     for pid, nodes in enumerate(partition_dict):
    #         partition_array[list(nodes)] = pid
    #     assert np.all((partition_array >= 0) & (partition_array < self._partition_num))
    #     return partition_array
    #
    # def _asyn_fluidc(self, ng: nx.DiGraph,num: int) -> np.ndarray:
    #     ng = ng.to_undirected()
    #     partition_dict = nx.community.asyn_fluidc(ng,num,max_iter=200,seed=123)
    #     partition_array = np.full(ng.number_of_nodes(), -1, dtype=np.int32)
    #     print(f"fluidc get {len(partition_dict)} communities")
    #     for pid, nodes in enumerate(partition_dict):
    #         partition_array[list(nodes)] = pid
    #     assert np.all((partition_array >= 0) & (partition_array < self._partition_num))
    #     return partition_array


    # def _lukes_partitioning(self,ng: nx.DiGraph, num: int) -> np.ndarray:
    #     ng = ng.to_undirected()
    #     partition_list = nx.community.lukes_partitioning(ng, "weight", cutoff=num, best_n=num)
    #     partition_array = np.full(ng.number_of_nodes(), -1, dtype=np.int32)
    #     for pid, p in enumerate(partition_list):
    #         partition_array[p] = pid
    #
    #     assert np.all((partition_array >= 0) & (partition_array < self._partition_num))
    #     return partition_array


class Scheduler():
    def __init__(self, partition_num, relation_martrix, parallel_partition_num, threshold, edge_num):
        self.partition_num = partition_num
        self.relation_matrix = relation_martrix
        self.threshold = threshold
        self.parallel_partition_num = parallel_partition_num
        self.edge_num = edge_num
        self.balance_edge_num = math.ceil(threshold * edge_num)

    def compute_relation_score(self, pid, plan):
        scores = [self.relation_matrix[pid][p] + self.relation_matrix[p][pid] for p in plan]
        print(f"partition {pid} with plan",end=" ") 
        print(plan,end=" ")
        print(f"relation score: {sum(scores):.4f}")
        return sum(scores)

    def generate_plan(self):
        """
        greedy base method to select partitions that are most related
        """
        scheduled = [0] * self.partition_num
        train_plan = []
        for pid in range(self.partition_num):
            if scheduled[pid] == 1:
                continue
            plan = []
            plan_score = 0
            plan.append(pid)
            plan_edge_num = self.relation_matrix[pid][pid]
            for k in range(self.parallel_partition_num - 1):
                best_pid = -1
                best_relation_score = 0
                for j in range(pid + 1, self.partition_num):
                    if scheduled[j] == 0:
                        relation_score = self.compute_relation_score(j, plan)
                        if relation_score > best_relation_score:
                            best_pid = j
                            best_relation_score = relation_score
                # if best_pid != -1 and self.threshold is not None and plan_score + best_relation_score < self.threshold:
                if best_pid != -1 and plan_edge_num + self.relation_matrix[pid][best_pid] + self.relation_matrix[best_pid][pid] < self.balance_edge_num:
                    scheduled[best_pid] = 1
                    plan.append(best_pid)
                    plan_score += best_relation_score
                    print(f"add partition {best_pid} to plan {plan} with score {best_relation_score}")
                else:
                    break
            train_plan.append(plan)

        print(f"generated train plan :")
        for plan in train_plan:
            print(plan)

        return train_plan


class MergePartition:
    def __init__(self, partitions:List[Partition], add_cross_partition:bool=True):
        all_edges = []
        pids = [p.id() for p in partitions]
        for p in partitions:
            for pid2 in pids:
                if p.id() == pid2:
                    all_edges.extend(p.edges())
                else:
                    all_edges.extend(p.cut_edges(pid2))

        all_edges = sorted(all_edges,key=lambda e: e.ets)
        self.srcs = [e.src for e in all_edges]
        self.dsts = [e.dst for e in all_edges]
        self.etss = [e.ets for e in all_edges]
        self.eids = [e.eid for e in all_edges]

    def __len__(self):
        return len(self.srcs)


class MergePartitionLoader:
    def __init__(self, merge_partition,batch_size):
        self.merge_partition = merge_partition
        self.batch_size = batch_size
        self.size = len(self.merge_partition)
        self.cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt < self.size:
            start = self.cnt
            end = min(self.size, self.cnt + self.batch_size)
            self.cnt = end
            return (self.merge_partition.srcs[start:end],
                    self.merge_partition.dsts[start:end],
                    self.merge_partition.etss[start:end],
                    self.merge_partition.eids[start:end])
        else:
            raise StopIteration




class PPDataset(Dataset):
    def __init__(self, merge_partitions:List[MergePartition]):
        super().__init__()
        self.merge_partitions = merge_partitions
        self.size = max([len(mp) for mp in self.merge_partitions])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.merge_partitions.edges[idx]

    def mp_num(self):
        return len(self.merge_partitions)

    def iter_edges(self, batch_size):
        mp_iters = [MergePartitionLoader(mp, batch_size) for mp in self.merge_partitions]
        batch_results = [torch.empty(0) for _ in range(4)]
        srcs, dsts, etss, eids = [], [], [], []

        for mp_iter in mp_iters:
            try:
                batch = next(mp_iter)
                srcs.extend(batch[0])
                dsts.extend(batch[1])
                etss.extend(batch[2])
                eids.extend(batch[3])
            except StopIteration:
                continue

        batch_results[0] = np.array(srcs,dtype=np.int32)
        batch_results[1] = np.array(dsts,dtype=np.int32)
        batch_results[2] = np.array(eids,dtype=np.int32)
        batch_results[3] = np.array(etss,dtype=np.float32)

        if len(batch_results[0]) == 0:
            raise StopIteration

        yield batch_results

# class PPDataLoader():
#     def __init__(self, merge_partitions, batch_size):
#         self.batch_size = batch_size
#         self.merge_partitions = merge_partitions
#
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#
#         batch_results = [torch.empty(0) for _ in range(4)]
#         srcs = []
#         dsts = []
#         etss = []
#         eids = []
#         for mp in self.merge_partitions:
#             try:
#                 batch = next(mpiter)
#                 srcs.append(batch[0])
#                 dsts.append(batch[1])
#                 etss.append()
#             except StopIteration:
#                 continue
#
#         if batch_results[0].size(0) == 0:
#             raise StopIteration
#
#         return batch_results
#




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--minp',type=int, default=2)
    parser.add_argument('--maxp',type=int, default=10)
    parser.add_argument('--alg',type=str,default="metis")
    args = parser.parse_args()

    minp = args.minp
    maxp = args.maxp
    data = args.data
    alg = args.alg

    # g = KarateClubDataset()[0]
    # indptr, indices, eid = g.adj_tensors('csr')
    # if len(eid) == 0:
    #     eid = [i for i in range(len(indices))]
    # ets = [i + 1 for i in range(len(indices))]
    if data in ["WIKI","REDDIT","MOOC","MAG","LASTFM","GDELT"]:
        g = np.load('DATA/{}/int_train.npz'.format(data))
        tcsr = TCSR(g["indptr"], g["indices"], g["eid"], g["ts"])
        num_edge = len(g["ts"])
        num_nodes = len(g["indptr"]) - 1
        print(num_nodes, num_edge)
        exit()
    elif data.find("amazon") != -1: # amazon
        g, _, _ = amazon_dataset("DATA/{}.txt".format(data))
        tcsr = TCSR.from_dglgraph(g)
        num_edge = len(tcsr.eid)
    elif data == "orkut":
        g, _, _ = orkut_dataset()
        tcsr = TCSR.from_dglgraph(g)
        num_edge = len(tcsr.eid)
    elif data in ["gplus","twitter","facebook","pokec"]:
        g, _, num_edge = social_dataset(data)
        tcsr = TCSR.from_dglgraph(g)
    elif data == "livejournal":
        g, _, num_edge = livejournal_dataset()
        tcsr = TCSR.from_dglgraph(g)
    elif data in ["BerkStan","Google","Stanford"]:
        g, _, num_edge = web_dataset(data)
        tcsr = TCSR.from_dglgraph(g)
    elif data in ["HepPh","Patents"]:
        g, _, num_edge = cite_dataset(data)
        tcsr = TCSR.from_dglgraph(g)
    else:
        raise NotImplementedError


    # indptr = indptr.tolist()
    # indices = indices.tolist()
    # indptr  = [0,1,2,6]
    # indice = [1,2,0,0,0,2]
    # eid = [0,1,2,3,4,5]
    # ets = [5,16,15,21,30,35]
    # tcsr = TCSR(indptr, indices, eid, ets)
    plist = [i for i in range(2,16)]
    if data == "MAG":
        plist += [i for i in range(20,101,5)] 
    for p in plist:
        print(f"test with partition num {p}")
        pg = PartitionedGraph(tcsr, p)
        pg.prepare_graph(alg=alg)
        import math
        threshold = math.floor(0.01 * num_edge)
        if p < 20:
            plan = pg.create_train_plan(3, threshold, False)
        else:
            plan = pg.create_train_plan(5, threshold, False)

    # ps = pg.partitions
    # loader = iter(DataLoader(p, batch_size=1000))
    # for batch in loader:
    #     print(batch[0].shape)
    # next(loader)
    # ppd = PPDataset(ps[0], ps[1], ps[2])
    # pploader = ppd.iter_edges(3000)
    # for batchs in iter(pploader):
    #     print(len(batchs), batchs[0].shape)

    # print(g.adj())
    # print(f"metis graph weight {g.edata['weight']}")
    # print(f"metis graph {g}")
    # indptr, indices, _ = g.adj_tensors("csr")
    # indptr, indices = indptr.tolist(), indices.tolist()
    # eweights = g.edata["weight"].tolist()
    # # print(indptr,indices,eweights)
    # n_cut, partition_array = pymetis.part_graph(2,xadj=indptr,adjncy=indices,eweights=eweights)
    # print(f"edge cut: {n_cut}")
    # print(f"partition array {partition_array}")
    # src, dst = g.edges()
    # removed_src = []
    # removed_dst = []
    # for i in range(len(src)):
    #     if partition_array[src[i]] != partition_array[dst[i]]:
    #         removed_src.append(src[i])
    #         removed_dst.append(dst[i])
    # eids = g.edge_ids(removed_src,removed_dst)
    # g.remove_edges(eids)
    # print(f"partition graph {g}")
    # ng = dgl.to_networkx(g,edge_attrs=["weight"])
    # plt.figure(figsize=(15,7))
    # nx.draw(ng)
    # plt.draw()
