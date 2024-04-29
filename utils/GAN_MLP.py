import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import *
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from deepadr.Transformer import Transformer
from .conv import GNN_node, GNN_node_Virtualnode
from .dataset import create_setvector_features
from .model_attn_siamese import FeatureEmbAttention as GNNLayerEmbAttention

from torch_scatter import scatter_mean



class DrugMLP(nn.Module):
    def __init__(self, num_layer=5, emb_dim=100,JK="multilayer",
                 gnn_type='gat', residual=False, drop_ratio=0.5):
        super(DrugMLP, self).__init__()
        self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK,drop_ratio=drop_ratio, residual=residual,
                                 gnn_type=gnn_type)
        self.pool = global_mean_pool
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)

    def forward(self, x, edge_index, batch):
        # print("x:"+str(x.shape))
        h_node = self.gnn_node(x, edge_index, None)

        h_graphs = [self.pool(h, batch) for h in h_node]
        # print(h_graphs[0].shape)

        h_graph_cat = torch.cat(h_graphs, dim=1)
        # print(h_graph_cat.shape)
        x = torch.relu(self.fc1(h_graph_cat))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层通常不加激活函数
        return x
