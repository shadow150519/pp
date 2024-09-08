from dataclasses import dataclass
from typing import Dict, Set, Tuple, List, DefaultDict
import sys
import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, parent_dir)
import torch
import dgl
from collections import defaultdict


class TCSR():
    def __init__(self, indptr, indice, eid, ets) -> None:
        self.ind = indptr
        self.nbr = indice
        self.eid = eid
        self.ets = ets

    def __len__(self):
        return len(self.eid)

    @classmethod
    def from_dglgraph(cls, g: dgl.DGLGraph, temporal=False):
        indptr, indices, eid = g.adj_tensors("csr")
        if eid is None:
            eid = torch.arange(0, g.num_edges(), dtype=torch.int32)
        ets = None
        if not temporal:
            ets = torch.zeros(g.num_edges(), dtype=torch.int32)
        return cls(indptr, indices, eid, ets)


@dataclass(frozen=True)
class TEdge:
    src: int
    dst: int
    ets: int
    eid: int


class Partition:
    def __init__(self, id: int, partition_num: int):
        self._id = id
        self._partition_num = partition_num
        self._edges = []
        # TODO: edge_list的类型是什么
        self._edge_list: DefaultDict[int, Set[TEdge]] = defaultdict(set)
        self._edge_cuts: List[DefaultDict[int, Set[TEdge]]] = [defaultdict(set) for _ in range(partition_num)]
        self._node_set: Set[int] = set()

    def add_edge(self, e: TEdge):
        self._edges.append(e)
        self._edge_list[e.src].add(e)

    def add_node(self, node_id: int):
        self._node_set.add(node_id)

    def add_cut_edge(self, e: TEdge, dst_partition_id: int):
        self._edge_cuts[dst_partition_id][e.src].add(e)

    def get_neighbors(self, id: int) -> Set[TEdge]:
        return self._edge_list.get(id, set())

    def nodes(self):
        return self._node_set

    def edges(self):
        return self._edges

    def cut_edges(self, pid=None):
        cedges = []
        for cut_edge_set in self._edge_cuts[pid].values():
            cedges.extend(list(cut_edge_set))

        return cedges

    def id(self):
        return self._id

    def node_num(self):
        return len(self._node_set)

    def edge_num(self):
        return len(self._edges)

    def cut_edge_num(self, pid=None):
        num = 0
        if pid is None:
            for p in self._edge_cuts:
                num += sum([len(s) for s in p.values()])
        else:
            p = self._edge_cuts[pid]
            num = sum([len(s) for s in p.values()])
        return num

    def exist(self, id: int):
        return id in self._node_set

    def to_tcsr(self) -> TCSR:
        pass
