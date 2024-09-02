import dgl
import torch
from dgl import DGLGraph
from typing import Tuple


def amazon_dataset(path: str) -> tuple[DGLGraph, int, int]:
    with open(path, "r") as f:
        lines = f.readlines()
        num_node, num_edge = int(lines[2].split()[2]), int(lines[2].split()[4])
        src = []
        dst = []
        for i in range(4, len(lines)):
            s, d = int(lines[i].split()[0]), int(lines[i].split()[1])
            src.append(s)
            dst.append(d)
        g = dgl.graph((src, dst))
    return g, num_node, num_edge

def orkut_dataset() -> tuple[DGLGraph, int, int]:
    node_map = {}
    node_set = set()
    src, dst = [], []
    num_edge, num_node = 0, 0
    gid = 0
    with open("DATA/com-orkut.ungraph.txt","r") as f:
        for i, line in enumerate(f):
            if i in [0, 1, 3]:
                continue
            if i == 2:
                num_node, num_edge = int(line.split()[2]), int(line.split()[4])
                continue
            s, d = int(line.split()[0]), int(line.split()[1])
            if s not in node_set:
                node_set.add(s)
                node_map[s] = gid
                gid += 1
            if d not in node_set:
                node_set.add(d)
                node_map[d] = gid
                gid += 1
            src.append(node_map[s])
            dst.append(node_map[d])
            # num_edge += 1
    g = dgl.graph((src,dst),num_nodes=num_node)
    return g, num_node, num_edge

def social_dataset(name:str) -> tuple[DGLGraph, int, int]:
    node_map = {}
    node_set = set()
    src, dst = [], []
    num_edge = 0
    gid = 0
    with open("DATA/{}.txt".format(name),"r") as f:
        for i, line in enumerate(f):
            s, d = int(line.split()[0]), int(line.split()[1])
            if s not in node_set:
                node_set.add(s)
                node_map[s] = gid
                gid += 1
            if d not in node_set:
                node_set.add(d)
                node_map[d] = gid
                gid += 1
            src.append(node_map[s])
            dst.append(node_map[d])
            num_edge += 1
    g = dgl.graph((src, dst), num_nodes=len(node_set))
    return g, len(node_set), num_edge

def livejournal_dataset()-> tuple[DGLGraph, int, int]:
    node_map = {}
    node_set = set()
    src, dst = [], []
    num_edge, num_node = 0, 0
    gid = 0
    with open("DATA/soc-LiveJournal1.txt","r") as f:
        for i, line in enumerate(f):
            if i in [0, 1, 3]:
                continue
            if i == 2:
                num_node, num_edge = int(line.split()[2]), int(line.split()[4])
                continue
            s, d = int(line.split()[0]), int(line.split()[1])
            if s not in node_set:
                node_set.add(s)
                node_map[s] = gid
                gid += 1
            if d not in node_set:
                node_set.add(d)
                node_map[d] = gid
                gid += 1
            src.append(node_map[s])
            dst.append(node_map[d])
            # num_edge += 1
    g = dgl.graph((src,dst),num_nodes=num_node)
    return g, num_node, num_edge

def sx_dataset(name:str)-> tuple[DGLGraph, int, int]:
    node_map = {}
    node_set = set()
    src, dst, ets = [], [], []
    num_edge, num_node = 0, 0
    gid = 0
    with open("DATA/sx-{}.txt".format(name),"r") as f:
        for i, line in enumerate(f):
            s, d, ts = int(line.split()[0]), int(line.split()[1]), int(line.split()[2])
            if s not in node_set:
                node_set.add(s)
                node_map[s] = gid
                gid += 1
            if d not in node_set:
                node_set.add(d)
                node_map[d] = gid
                gid += 1
            src.append(node_map[s])
            dst.append(node_map[d])
            ets.append(ts)
            num_edge += 1
    num_node = len(node_set)
    g = dgl.graph((src,dst),num_nodes=num_node)
    g.edata["ets"] = torch.tensor(ets,dtype=torch.int)
    return g, num_node, num_edge

def cite_dataset(name:str):
    node_map = {}
    node_set = set()
    src, dst = [], []
    num_edge, num_node = 0, 0
    gid = 0
    with open("DATA/cit-{}.txt".format(name),"r") as f:
        for i, line in enumerate(f):
            if i in [0, 1, 3]:
                continue
            if i == 2:
                num_node, num_edge = int(line.split()[2]), int(line.split()[4])
                continue
            s, d = int(line.split()[0]), int(line.split()[1])
            if s not in node_set:
                node_set.add(s)
                node_map[s] = gid
                gid += 1
            if d not in node_set:
                node_set.add(d)
                node_map[d] = gid
                gid += 1
            src.append(node_map[s])
            dst.append(node_map[d])
            # num_edge += 1
    g = dgl.graph((src,dst),num_nodes=num_node)
    return g, num_node, num_edge

def web_dataset(name:str):
    node_map = {}
    node_set = set()
    src, dst = [], []
    num_edge, num_node = 0, 0
    gid = 0
    with open("DATA/web-{}.txt".format(name),"r") as f:
        for i, line in enumerate(f):
            if i in [0, 1, 3]:
                continue
            if i == 2:
                num_node, num_edge = int(line.split()[2]), int(line.split()[4])
                continue
            s, d = int(line.split()[0]), int(line.split()[1])
            if s not in node_set:
                node_set.add(s)
                node_map[s] = gid
                gid += 1
            if d not in node_set:
                node_set.add(d)
                node_map[d] = gid
                gid += 1
            src.append(node_map[s])
            dst.append(node_map[d])
            # num_edge += 1
    g = dgl.graph((src,dst),num_nodes=num_node)
    return g, num_node, num_edge



if __name__ == "__main__":
    # g,n,e = amazon_dataset("DATA/amazon0302.txt")
    # g, n, e = orkut_dataset()
    # g, n, e = social_dataset("DATA/facebook.txt")
    # g, n, e = social_dataset("DATA/twitter.txt")
    #g, n, e = social_dataset("DATA/gplus.txt")
    # g, n, e = social_dataset("DATA/pokec.txt")
    # g, n, e = livejournal_dataset()
    # g, n, e = sx_dataset("askubuntu")
    # g, n, e = sx_dataset("stackoverflow")
    # g, n, e = cite_dataset("HepPh")
    # g, n, e = cite_dataset("Patents")
    #g, n, e = web_dataset("BerkStan")
    g, n, e = web_dataset("Google")
    #g, n, e = web_dataset("Stanford")
    print(g,n,e)
