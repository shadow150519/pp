from typing import Union

import numpy
import numpy as np
import torch
import sys
import os
from . import libgpucache as gpucache
import math
import time

dtype_size_map = {
    torch.int32: 4,
    torch.int64: 8,
    torch.float32: 4,
    torch.float64: 8
}

strategy_map = {
    "LFU": gpucache.CacheConfig.LFU,
    "LRU": gpucache.CacheConfig.LRU,
    "FIFO": gpucache.CacheConfig.FIFO
}

INT32_MAX = 0xFFFFFFFF


class CacheWrapper():
    def __init__(self, embedding_table: torch.Tensor, cache_size, strategy, embedding_dim, max_query,
                 device_id,name=None) -> None:
        self.embedding = embedding_table
        self.cache_size = cache_size
        value_size = dtype_size_map[embedding_table.dtype] * embedding_dim
        self.strategy = strategy_map[strategy]
        cfg = gpucache.CacheConfig(self.strategy, cache_size, 8 if cache_size > INT32_MAX else 4, value_size,
                                   max_query, device_id, embedding_dim)
        self.device_id = device_id
        self.cache = gpucache.NewCache(embedding_table, cfg)
        self.kdtype = torch.int32 if cache_size <= INT32_MAX else torch.int64
        self.vdtype = embedding_table.dtype
        self.max_query_num = max_query
        self.embedding_dim = embedding_dim
        self.get_time = 0.0
        self.get_call_num = 0
        self.put_time = 0.0
        self.put_call_num = 0
        self.query_num = 0
        self.hit_num = 0
        self.name = name

    def Get(self, querys: Union[list, tuple, torch.Tensor, numpy.ndarray]):
        # start_time = time.time()
        if isinstance(querys, list) or isinstance(querys, tuple):
            querys = torch.tensor(querys, dtype=self.kdtype, device=torch.device("cuda", self.device_id))
        elif isinstance(querys, np.ndarray):
            querys = torch.from_numpy(querys).to(torch.device("cuda", self.device_id), dtype=self.kdtype)
        elif isinstance(querys, torch.Tensor):
            querys = querys.to(torch.device("cuda", self.device_id), dtype=self.kdtype)
            assert querys.device == torch.device("cuda",self.device_id)
        #assert self.cache.MaxQueryNum() >= querys.shape[0]
        query_num = querys.shape[0]
        values = torch.empty(query_num,self.embedding_dim, dtype=self.vdtype,device=torch.device('cuda',self.device_id))
        total_n_missing = 0
        for start in range(0, query_num,self.max_query_num):
            self.get_call_num += 1
            end = min(query_num,start+self.max_query_num)
            values[start:end], missing_indexs, missing_keys, n_missing = self.cache.Get(end-start, querys[start:end])
            if n_missing.item() > 0:
                missing_indexs = missing_indexs.long() + start
                values[missing_indexs,:] = self.embedding.index_select(0, missing_keys.cpu()).cuda(self.device_id).reshape(n_missing,-1)
                # values[missing_indexs.long()] = self.embedding[missing_keys.cpu().long()].cuda(self.device_id)
                self.Put(missing_keys.shape[0], missing_keys, values.index_select(0, missing_indexs))
                total_n_missing += n_missing.item()
                self.put_call_num += 1
        self.hit_num += (querys.shape[0] - total_n_missing)
        self.query_num += querys.shape[0]
            

        return values

    def reset_cache(self):
        self.get_call_num, self.put_call_num = 0, 0
        self.get_time, self.put_time = 0.0, 0.0
        self.hit_num = 0
        self.query_num = 0

    def report_time(self):
        print(f"cache statistics:")
        print(f"\tname {self.name}")
        print(f"\tcall get {self.get_call_num} times, total time {self.get_time}")
        print(f"\tcall put {self.put_call_num} times, total time {self.put_time}")
        print(f"\tquery {self.query_num} times, hit {self.hit_num} times, hit rate {(self.hit_num / self.query_num):.3f}")
        return

    def Put(self, kv_num, put_keys: torch.Tensor, put_values: torch.Tensor):
        assert put_keys.get_device() == put_values.get_device() == self.device_id
        assert put_keys.dtype == self.kdtype
        self.cache.Put(put_keys.shape[0], put_keys, put_values)
        return


if __name__ == "__main__":
    embedding = torch.arange(0, 300, dtype=torch.float32, device="cuda:0").reshape(-1, 100)
    cache = CacheWrapper(embedding, 4096, "LRU", 100, 1024, 0)
