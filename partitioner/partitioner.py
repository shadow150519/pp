import numpy as np
import pandas as pd
from enum import Enum
from partition2 import Partition, TEdge
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from partition import TCSR
from typing import List, Tuple, Set, Dict

class PartitionAlg(Enum):
    LDG=1
    METIS=2
    
class Partitioner:
    def __init__(self, dataset:str,partition_num:int,alg:PartitionAlg=PartitionAlg.LDG):
        self._partition_num = partition_num
        self._dataset = dataset
        self._partition_alg = alg
        self._partitions: List[Partition] = [Partition(id,partition_num) for id in range(partition_num)]
        self._origin_graph = pd.read_csv("DATA/{}/edges.csv".format(dataset))
        self._train_graph = self._origin_graph[self._origin_graph["ext_roll"] == 0]
        train_g = np.load('DATA/{}/int_train.npz'.format(dataset)) # train_g
        self._tcsr = TCSR(train_g["indptr"], train_g["indices"], train_g["eid"], train_g["ts"])
        self._node_num = len(train_g["indptr"]) - 1
        self._edge_num = len(train_g["indices"])
        self._cut_edge_num = 0
    
    
    def partition_graph(self):
        if self._partition_alg == PartitionAlg.LDG:
            return self.ldg_partition(self._train_graph, self._partition_num)
        

    def ldg_partition(self, edges:pd.DataFrame, num:int)-> Tuple[List[Partition],np.ndarray]:
        def get_intersection_edge_num(tcsr:TCSR, p:Partition, node_id:int):
            score = 0
            pnodes = p.nodes()
            for i in range(tcsr.ind[node_id],tcsr.ind[node_id + 1]):
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
                    self._partitions[partition_array[src]].add_edge(TEdge(src,dst,eid,ets))
                else:
                    self._partitions[partition_array[src]].add_cut_edge(TEdge(src,dst,ets,eid), partition_array[dst])
                    self._cut_edge_num += 1
                continue
            
            partition_score_src = np.zeros(self._partition_num,dtype=np.float32)
            partition_score_dst = np.zeros(self._partition_num,dtype=np.float32)
            
            # src or/and dst aren't partitioned
            for pid, partition in enumerate(self._partitions):
                partition_node_num = partition.node_num()
                src_score, dst_score = 0.0, 0.0
                
                # TODO: node balance or edge balance or non balance
                # need to relax constraint ?
                BALANCE_FACTOR = 1
                partition_weight = (1 - (partition_node_num / ((total_node_num + 1e-6) / (self._partition_num * BALANCE_FACTOR))))
                
                if partition_array[src] == -1:
                    src_score = min(get_intersection_edge_num(self._tcsr,partition,src),1) * partition_weight
                if partition_array[dst] == -1:
                    dst_score = min(get_intersection_edge_num(self._tcsr,partition,dst),1) * partition_weight
                
                partition_score_src[pid] = src_score
                partition_score_dst[pid] = dst_score
                         
                
            if partition_array[src] == -1:
                partition_array[src] = partition_score_src.argmax()
                self._partitions[partition_array[src]].add_node(src)
                print(f"add node {src} to partition {partition_array[src]} with score {partition_score_src[partition_array[src]]}")
                total_node_num += 1
            if partition_array[dst] == -1:
                partition_array[dst] = partition_score_dst.argmax()
                self._partitions[partition_array[dst]].add_node(dst)
                print(f"add node {dst} to partition {partition_array[dst]} with score {partition_score_dst[partition_array[dst]]}")
                total_node_num += 1
            if partition_array[src] == partition_array[dst]:
                self._partitions[partition_array[src]].add_edge(TEdge(src,dst,eid,ets))
            else:
                self._partitions[partition_array[src]].add_cut_edge(TEdge(src,dst,ets,eid), partition_array[dst])
                self._cut_edge_num += 1
                
        return self._partitions, partition_array
    
    def print_stats(self):
        print(f"\tdataset name: {self._dataset} partition num: {self._partition_num} partition alg: {self._partition_alg}")      
        print(f"\tnode num: {self._node_num} edge num: {self._edge_num}") 
        print(f"\tcut edge num: {self._cut_edge_num} cut_ratio: {self._cut_edge_num/self._edge_num:.3f}")        
        print(f"\trelation matrix")
        for i in range(self._partition_num):
            for j in range(self._partition_num):
                if i == j:
                    print(self._partitions[i].edge_num(), end=" ")
                else:
                    print(self._partitions[i].cut_edge_num(j), end=" ")
            print()
        for i, partition in enumerate(self._partitions):
            print(f"\tpartition {i} stats:")
            print(f"\t\tnode num: {partition.node_num()} edge num: {partition.edge_num()}")
            print(f"\t\tcut edge: {partition.cut_edge_num()} cut ratio: {partition.cut_edge_num()/(partition.edge_num() + partition.cut_edge_num()):.3f}")            
                
        
if __name__ == "__main__":
    partitioner = Partitioner("WIKI",10)
    partitioner.partition_graph()
    partitioner.print_stats()