
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


class GNN(torch.nn.Module):

    def __init__(self, num_layer=5, emb_dim=300,
                 gnn_type='gat', virtual_node=True, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 with_edge_attr=False):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        
        self.graph_pooling = graph_pooling
        self.with_edge_attr = with_edge_attr
        self.layer_pooling = GNNLayerEmbAttention(emb_dim)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                                 gnn_type=gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                     gnn_type=gnn_type, with_edge_attr=self.with_edge_attr)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        
        else:
            raise ValueError("Invalid graph pooling type.")



    def forward(self, x, edge_index, edge_attr, batch):
        # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        # print("x:"+str(x.shape))
        if self.with_edge_attr:
            h_node = self.gnn_node(x, edge_index, edge_attr)
        else:
            # print("h_node:::")
            h_node = self.gnn_node(x, edge_index, None)
        # print(h_node[1].shape)
        if self.JK == "multilayer":
            h_graphs = [self.pool(h, batch) for h in h_node]
            # print(h_graphs[0].shape)

            h_graph_cat = torch.cat(h_graphs, dim=1)
            # print(".....")
            # print(h_graph_cat.shape)

            h_graph_t = h_graph_cat.reshape(h_graph_cat.shape[0], len(h_graphs), h_graph_cat.shape[1] // len(h_graphs))

            h_graph, layer_weights = self.layer_pooling(h_graph_t)

        else:
            h_graph = self.pool(h_node, batch)

        #         return self.graph_pred_linear(h_graph)
        return h_graph


class GATNet(torch.nn.Module):

    def __init__(self, num_features_xd=9, n_output=2, output_dim=128, dropout=0.2, heads=10, file=None):
        super(GATNet, self).__init__()


        # graph drug layers
        self.drug1_gcn1 = GATConv(num_features_xd, output_dim, heads=10, dropout=dropout)
        self.drug1_gcn2 = GATConv(output_dim * 10, output_dim, dropout=dropout)
        self.drug1_fc_g1 = nn.Linear(output_dim, output_dim)
        
        
        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim


    def forward(self, x, edge_index, edge_attr, batch):
        
        x1 = self.drug1_gcn1(x, edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2)

        x1 = self.drug1_gcn2(x1, edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2)

        x1 = global_max_pool(x1, batch)         # global max pooling


        x1 = self.drug1_fc_g1(x1)
        x1 = self.relu(x1)

        return x1

def _init_model_params(named_parameters):
    for p_name, p in named_parameters:
        param_dim = p.dim()
        if param_dim > 1: # weight matrices
            nn.init.xavier_uniform_(p)
        elif param_dim == 1: # bias parameters
            if p_name.endswith('bias'):
                nn.init.uniform_(p, a=-1.0, b=1.0)

class DeepAdr_SiameseTrf(nn.Module):

    def __init__(self, input_dim, dist, expression_dim, gene_embed_dim=1, num_classes=2, drop=0.5, do_softmax=True):
        
        super().__init__()
        
        self.do_softmax = do_softmax
        self.num_classes = num_classes
        
        if dist == 'euclidean':
            self.dist = nn.PairwiseDistance(p=2, keepdim=True)
            self.alpha = 0
        elif dist == 'manhattan':
            self.dist = nn.PairwiseDistance(p=1, keepdim=True)
            self.alpha = 0
        elif dist == 'cosine':
            self.dist = nn.CosineSimilarity(dim=1)
            self.alpha = 1

        self.Wy = nn.Linear(2*input_dim+1, num_classes)
        
        self.Wy_ze = nn.Linear(2*input_dim+1+expression_dim, input_dim)
        self.Wy3 = nn.Linear(input_dim, num_classes)
        
        self.drop = nn.Dropout(drop)
        # perform log softmax on the feature dimension
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._init_params_()
        print('updated')
        print('num classes:', self.num_classes)
        
        
    def _init_params_(self):
        _init_model_params(self.named_parameters())
    
    def forward(self, Z_a, Z_b, Z_e=None):
        """
        Args:
            Z_a: tensor, (batch, embedding dim)
            Z_b: tensor, (batch, embedding dim)
        """

        dist = self.dist(Z_a, Z_b).reshape(-1,1)
        # update dist to distance measure if cosine is chosen
        dist = self.alpha * (1-dist) + (1-self.alpha) * dist
        
        if (Z_e is not None):
            out = torch.cat([Z_a, Z_b, dist, Z_e], axis=-1)
            y = self.Wy_ze(out)
            y = self.drop(y)
            y = self.Wy3(y)
        else:
            out = torch.cat([Z_a, Z_b, dist], axis=-1)
            y = self.Wy(out)
            
        if (self.num_classes == 0):
            return out, dist
        
        
        if self.do_softmax:
            return self.log_softmax(y), dist
        else:
            return y, dist

class DeepSynergy(nn.Module):
    def __init__(self, D_in=2636, H1=8192, H2=4096, D_out=2, drop=0.5):
        super(DeepSynergy, self).__init__()

        # self.Transformer = Transformer()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(D_in, H1) # Fully Connected
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)
        self.drop_in = nn.Dropout(drop)
        self.drop = nn.Dropout(drop)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.batch_norm1 = nn.BatchNorm1d(H1)
        self.batch_norm2 = nn.BatchNorm1d(H2)
        self._init_weights()
        
        print(self.drop, self.drop_in)

    def forward(self, x):

        # x = x.reshape(-1,1,696)
        # x = self.Transformer(x)
        # x = x.reshape(-1,696)

        x =self.fc1(x)
        # print(x.shape)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.drop_in(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        return self.log_softmax(x)

    def _init_weights(self):
        for m in self.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.uniform_(-1,0)
    # def _init_weights(self):
    #     for m in self.modules():
    #         if(isinstance(m,Transformer)):
    #             continue
    #         if(isinstance(m, nn.Linear)):
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.uniform_(-1, 0)

    
class ExpressionNN(nn.Module):
    def __init__(self, D_in=926, H1=8192, H2=4096, D_out=2, drop=0.5):
        super(ExpressionNN, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(D_in, H1) # Fully Connected
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)
        self.drop_in = nn.Dropout(0.2)
        self.drop = nn.Dropout(drop)
#         self.log_softmax = nn.LogSoftmax(dim=-1)
        self._init_weights()
        
        print(self.drop, self.drop_in)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.drop_in(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
#         return self.log_softmax(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.uniform_(-1,0)
                
                
class DeepDDS_MLP(nn.Module):
    def __init__(self, D_in=(4*128), H1=2048, H2=512, H3=128, D_out=2, drop=0.2):
        super(DeepDDS_MLP, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(D_in, H1) # Fully Connected
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.out = nn.Linear(H3, D_out)
#         self.drop_in = nn.Dropout(0.2)
        self.drop = nn.Dropout(drop)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self._init_weights()
        
#         print(self.drop, self.drop_in)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.out(x)
        return self.log_softmax(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.uniform_(-1,0)