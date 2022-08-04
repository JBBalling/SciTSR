import torch
import os 
import numpy as np
import json

from scitsr.data.loader import TableDataset, TableInferDataset
from scitsr.eval import json2Table
from scitsr.model import GraphAttention
from scitsr.relation import Relation

path = "datasets"

hidden_size = 4
n_blocks = 3


dataset = TableInferDataset(path, exts=["chunk", "json"])
model = GraphAttention(dataset.n_node_features, dataset.n_edge_features, hidden_size, dataset.output_size, n_blocks)
model.load_state_dict(torch.load("gat-model.pt"))
model.eval()

list_of_rel = []

for j, ds in enumerate(dataset.dataset):
    output_rel = model(ds.nodes, ds.edges, ds.adj, ds.incidence)
    out = output_rel.cpu().detach().numpy()
    int_out = np.rint(out)
    

    for i, relation in enumerate(ds.relations):
        rel = Relation(ds.chunks[relation[0]].text, ds.chunks[relation[1]].text, np.where(int_out[i] == 1.)[0].item(), relation[0], relation[1])
        list_of_rel.append(rel)
        print(rel.__str__())

