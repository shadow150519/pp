import dgl
import torch
import numpy as np
import pandas as pd
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from partition import Partition, TEdge

from collections import defaultdict
import pymetis
import math
from torch.utils.data import Dataset, DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from partition import TCSR
from typing import List, Tuple, Set, Dict



class PartitionAlg(Enum):
    LDG = 1
    METIS = 2


class Partitioner:
    def __init__(self, dataset: str, partition_num: int, alg: PartitionAlg = PartitionAlg.METIS):
        self._partition_num = partition_num
        self._dataset = dataset
        self._partition_alg = alg
        self._partitions: List[Partition] = [Partition(id, partition_num) for id in range(partition_num)]
        self._origin_graph = pd.read_csv("DATA/{}/edges.csv".format(dataset))
        self._train_graph = self._origin_graph[self._origin_graph["ext_roll"] == 0]
        train_g = np.load('DATA/{}/int_train.npz'.format(dataset))  # train_g
        self._tcsr = TCSR(train_g["indptr"], train_g["indices"], train_g["eid"], train_g["ts"])
        self._node_num = len(train_g["indptr"]) - 1
        self._edge_num = len(train_g["indices"])
        self._cut_edge_num = 0

        # for metis, metis need undirected and simple graph
        self._metis_graph = None

        self._relation_matrix = np.zeros((partition_num, partition_num), dtype=np.float32)

    def partition_graph(self) -> Tuple[List[Partition], np.ndarray]:
        if self._partition_alg == PartitionAlg.LDG:
            partition_array = self.ldg_partition(self._train_graph, self._partition_num)
        elif self._partition_alg == PartitionAlg.METIS:
            partition_array = self.metis_partition()
        self.compute_relation_matrix()
        return self._partitions, partition_array

    def _to_metis_graph(self, tcsr: TCSR) -> dgl.DGLGraph:
        indptr, indices, eid = tcsr.ind, tcsr.nbr, tcsr.eid
        g = dgl.graph(('csr', (indptr, indices, eid)))
        g = dgl.to_simple(g, return_counts="weight")
        indptr, indices, _ = g.adj_tensors('csr')
        weight = g.edata["weight"]

        undirected_graph = defaultdict(int)
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

    def metis_partition(self) -> np.ndarray:
        self._metis_graph = self._to_metis_graph(self._tcsr)

        indptr, indices, eid = self._metis_graph.adj_tensors("csr")
        indptr, indices = indptr.tolist(), indices.tolist()
        # rows = []
        # for i in range(len(indptr) - 1):
        #     rows.extend([i] * (indptr[i + 1] - indptr[i]))
        # ids = self._metis_graph.edge_ids(rows, indices)
        if eid.nelement() == 0:
            eweights = self._metis_graph.edata["weight"]
        else:
            eweights = self._metis_graph.edata["weight"][eid]

        # eweights = self._metis_graph.edata["weight"].tolist()
        import time
        t_start = time.time()
        n_cut, partition_array = pymetis.part_graph(self._partition_num, xadj=indptr, adjncy=indices,
                                                    eweights=eweights)
        t_elapsed = time.time() - t_start
        print(f"metis partition time {t_elapsed}s")
        partition_array = np.array(partition_array)
        self._n_cut, self._partition_array = n_cut, partition_array

        for i in partition_array:
            self._partitions[partition_array[i]].add_node(i)

        indptr, indices, eid, ets = self._tcsr.ind, self._tcsr.nbr, self._tcsr.eid, self._tcsr.ets
        for row in range(len(indptr) - 1):
            for j in range(indptr[row],indptr[row+1]):
                if partition_array[row] == partition_array[indices[j]]:
                    self._partitions[partition_array[row]].add_edge(TEdge(src=row,dst=indices[j],ets=ets[j],eid=eid[j]))
                else:
                    self._partitions[partition_array[row]].add_cut_edge(TEdge(src=row, dst=indices[j], ets=ets[j], eid=eid[j]),partition_array[indices[j]])

        return partition_array

    def ldg_partition(self, edges: pd.DataFrame, partition_num: int) -> np.ndarray:
        def get_intersection_edge_num(tcsr: TCSR, p: Partition, node_id: int):
            score = 0
            pnodes = p.nodes()
            for i in range(tcsr.ind[node_id], tcsr.ind[node_id + 1]):
                if tcsr.nbr[i] in pnodes:
                    score += 1
            print(f"node {node_id} have {score} edges with partition {p.id()}")
            return score

        total_node_num = 0
        # total_edge_num = 0
        partition_array = np.full(self._node_num, -1)

        for idx, row in edges.iterrows():
            # print(row)
            src, dst, ets, eid = int(row["src"]), int(row["dst"]), int(row["time"]), int(row[0])

            # both src and dst are partitioned
            if partition_array[src] != -1 and partition_array[dst] != -1:
                if partition_array[src] == partition_array[dst]:
                    self._partitions[partition_array[src]].add_edge(TEdge(src, dst, eid, ets))
                else:
                    self._partitions[partition_array[src]].add_cut_edge(TEdge(src, dst, ets, eid), partition_array[dst])
                    self._cut_edge_num += 1
                continue

            partition_score_src = np.zeros(self._partition_num, dtype=np.float32)
            partition_score_dst = np.zeros(self._partition_num, dtype=np.float32)

            # src or/and dst aren't partitioned
            for pid, partition in enumerate(self._partitions):
                partition_node_num = partition.node_num()
                src_score, dst_score = 0.0, 0.0

                # TODO: node balance or edge balance or non balance
                # need to relax constraint ?
                BALANCE_FACTOR = 1
                partition_weight = (1 - (
                        partition_node_num / ((total_node_num + 1e-6) / (self._partition_num * BALANCE_FACTOR))))

                if partition_array[src] == -1:
                    src_score = min(get_intersection_edge_num(self._tcsr, partition, src), 1) * partition_weight
                if partition_array[dst] == -1:
                    dst_score = min(get_intersection_edge_num(self._tcsr, partition, dst), 1) * partition_weight

                partition_score_src[pid] = src_score
                partition_score_dst[pid] = dst_score

            if partition_array[src] == -1:
                partition_array[src] = partition_score_src.argmax()
                self._partitions[partition_array[src]].add_node(src)
                print(
                    f"add node {src} to partition {partition_array[src]} with score {partition_score_src[partition_array[src]]}")
                total_node_num += 1
            if partition_array[dst] == -1:
                partition_array[dst] = partition_score_dst.argmax()
                self._partitions[partition_array[dst]].add_node(dst)
                print(
                    f"add node {dst} to partition {partition_array[dst]} with score {partition_score_dst[partition_array[dst]]}")
                total_node_num += 1
            if partition_array[src] == partition_array[dst]:
                self._partitions[partition_array[src]].add_edge(TEdge(src, dst, eid, ets))
            else:
                self._partitions[partition_array[src]].add_cut_edge(TEdge(src, dst, ets, eid), partition_array[dst])
                self._cut_edge_num += 1

        return partition_array

    def compute_relation_matrix(self):
        for i in range(self._partition_num):
            for j in range(self._partition_num):
                if i == j:
                    self._relation_matrix[i][j] = self._partitions[i].edge_num()
                else:
                    self._relation_matrix[i][j] = self._partitions[i].cut_edge_num(j)
                    self._cut_edge_num += self._relation_matrix[i][j]
    def create_train_plan(self, parallel_partition_num=1, threshold=None, alg="pp1"):
        """
        :param parallel_partition_num: max parallel partition num
        :param threshold: for pp1, threshold is edges cross partition,for pp2, threshold is edge balance
        :param alg: pp1 or pp2
        :return: schedule plan
        """
        scheduler = Scheduler(self._partition_num, self._relation_matrix, self._edge_num, parallel_partition_num, threshold, alg)
        train_plan = scheduler.generate_plan()

        return train_plan

    def print_stats(self):
        print(
            f"\tdataset name: {self._dataset} partition num: {self._partition_num} partition alg: {self._partition_alg}")
        print(f"\tnode num: {self._node_num} edge num: {self._edge_num}")
        print(f"\tcut edge num: {self._cut_edge_num} cut_ratio: {self._cut_edge_num / self._edge_num:.3f}")
        print(f"\trelation matrix")
        for i in range(self._partition_num):
            for j in range(self._partition_num):
                if i == j:
                    print(f"{self._partitions[i].edge_num():6}", end=" ")
                else:
                    print(f"{self._partitions[i].cut_edge_num(j):6}", end=" ")
            print()
        for i, partition in enumerate(self._partitions):
            print(f"\tpartition {i} stats:")
            print(f"\t\tnode num: {partition.node_num()} edge num: {partition.edge_num()}")
            print(
                f"\t\tcut edge: {partition.cut_edge_num()} cut ratio: {partition.cut_edge_num() / (partition.edge_num() + partition.cut_edge_num()):.3f}")

    def edge_num(self):
        return self._edge_num

    def node_num(self):
        return self._node_num



class Scheduler():
    def __init__(self, partition_num, relation_martrix, edge_num, parallel_partition_num, threshold, alg):
        self.partition_num = partition_num
        self.relation_matrix = relation_martrix
        self.threshold = threshold
        self.parallel_partition_num = parallel_partition_num
        self.edge_num = edge_num
        self.balance_edge_num = threshold
        self.alg = alg
    def compute_relation_score(self, pid, plan):
        scores = [self.relation_matrix[pid][p] + self.relation_matrix[p][pid] for p in plan]
        print(f"partition {pid} with plan",end=" ")
        print(plan,end=" ")
        print(f"relation score: {sum(scores):.4f}")
        return sum(scores)
    def generate_plan(self):
        if self.alg == "pp1":
            train_plan = self.generate_plan_pp1()
        elif self.alg == "pp2":
            train_plan = self.generate_plan_pp2()
        else:
            raise NotImplementedError
        return train_plan
    def generate_plan_pp2(self):
        """
        greedy base method to select partitions that are most related
        """
        scheduled = [0] * self.partition_num
        train_plan = []
        import random
        partitions = [i for i in range(self.partition_num)]
        random.shuffle(partitions)
        for pid in partitions:
            if scheduled[pid] == 1:
                continue
            plan = []
            plan_score = 0
            plan.append(pid)
            plan_edge_num = self.relation_matrix[pid][pid]
            for k in range(self.parallel_partition_num - 1):
                best_pid = -1
                best_relation_score = 0
                for j in range(self.partition_num):
                    if scheduled[j] == 0:
                        relation_score = self.compute_relation_score(j, plan)
                        if relation_score > best_relation_score:
                            best_pid = j
                            best_relation_score = relation_score
                # if best_pid != -1 and self.threshold is not None and plan_score + best_relation_score < self.threshold:
                if best_pid != -1 and plan_edge_num + self.relation_matrix[best_pid][best_pid] + best_relation_score < self.balance_edge_num:
                    scheduled[best_pid] = 1
                    plan.append(best_pid)
                    plan_score += best_relation_score + self.relation_matrix[best_pid][best_pid]
                    print(f"add partition {best_pid} to plan {plan} with score {best_relation_score}")
                else:
                    break
            train_plan.append(plan)

        print(f"generated train plan :")
        for plan in train_plan:
            print(plan)

        return train_plan


    def generate_plan_pp1(self):
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
            for k in range(self.parallel_partition_num - 1):
                best_pid = -1
                best_relation_score = 1e6
                for j in range(pid + 1, self.partition_num):
                    if scheduled[j] == 0:
                        relation_score = self.compute_relation_score(j, plan)
                        if relation_score < best_relation_score:
                            best_pid = j
                            best_relation_score = relation_score
                if best_pid != -1 and (self.threshold is None or plan_score + best_relation_score < self.threshold):
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


class PP2Dataset(Dataset):
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
        finished = [0 for _ in range(len(mp_iters))]
        while True:
            batch_results = [torch.empty(0) for _ in range(4)]
            srcs, dsts, etss, eids = [], [], [], []
            for i in range(len(mp_iters)):
                if finished[i] == 1:
                    continue
                try:
                    batch = next(mp_iters[i])
                    srcs.extend(batch[0])
                    dsts.extend(batch[1])
                    etss.extend(batch[2])
                    eids.extend(batch[3])
                except StopIteration:
                    finished[i] = 1
                    continue

            batch_results[0] = np.array(srcs,dtype=np.int32)
            batch_results[1] = np.array(dsts,dtype=np.int32)
            batch_results[2] = np.array(eids,dtype=np.int32)
            batch_results[3] = np.array(etss,dtype=np.float32)

            if len(batch_results[0]) == 0:
                return

            yield batch_results


class PP1DataLoader:
    def __init__(self,partition:Partition,batch_size):
        self.partition = partition
        self.batch_size = batch_size
        self.size = len(self.partition)
        self.cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt < self.size:
            start = self.cnt
            end = min(self.size, self.cnt + self.batch_size)
            self.cnt = end
            return ([e.src for e in self.partition.edges()[start:end]],
                    [e.dst for e in self.partition.edges()[start:end]],
                    [e.ets for e in self.partition.edges()[start:end]],
                    [e.eid for e in self.partition.edges()[start:end]])
        else:
            raise StopIteration
class PP1Dataset(Dataset):
    def __init__(self, partitions:List[Partition]):
        self.partitions = partitions
        self.partition_num = len(partitions)
        self.size = max(p.edge_num() for p in partitions)

    def __len__(self):
        return self.size

    def iter_edges(self, bs):
        iters = [PP1DataLoader(p, batch_size=bs // self.partition_num) for p in self.partitions]
        finished = [0 for _ in range(len(iters))]
        while True:
            batch_results = [0 for _ in range(4)]
            srcs, dsts, etss, eids = [], [], [], []
            for i in range(len(iters)):
                if finished[i] == 1:
                    continue
                try:
                    batch = next(iters[i])
                    srcs.extend(batch[0])
                    dsts.extend(batch[1])
                    etss.extend(batch[2])
                    eids.extend(batch[3])
                except StopIteration:
                    finished[i] = 1
                    continue

            batch_results[0] = np.array(srcs, dtype=np.int32)
            batch_results[1] = np.array(dsts, dtype=np.int32)
            batch_results[2] = np.array(eids, dtype=np.int32)
            batch_results[3] = np.array(etss, dtype=np.float32)

            if len(batch_results[0]) == 0:
                return

            yield batch_results

class PP1DatasetChain():
    def __init__(self, pp1datasets:List[PP1Dataset]):
        self.pp1datasets = pp1datasets
        self.size = len(pp1datasets)

    def __len__(self):
        return self.size

    def iter_edges(self, bs):
        for pp1dataset in self.pp1datasets:
            pp1iter = pp1dataset.iter_edges(bs)
            for batch in pp1iter:
                yield batch




if __name__ == "__main__":
    partitioner = Partitioner("WIKI", 10)
    partitioner.partition_graph()
    partitioner.print_stats()
