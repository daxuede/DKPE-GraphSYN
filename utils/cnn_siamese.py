import torch
# from torch import nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import dgl
# import dgl.nn

class SH_SelfAttention(nn.Module):
    """ single head self-attention module
    """
    def __init__(self, input_size):
        
        super().__init__()
        # define query, key and value transformation matrices
        # usually input_size is equal to embed_size
        self.embed_size = input_size
        self.Wq = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wk = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wv = nn.Linear(input_size, self.embed_size, bias=False)
        self.softmax = nn.Softmax(dim=2) # normalized across feature dimension
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, deepadr similarity type vector, input_size)
        """
        X_q = self.Wq(X) # queries
        X_k = self.Wk(X) # keys
        X_v = self.Wv(X) # values
        
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4))
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        
        attn_w_normalized = self.softmax(attn_w)
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
                
        return z, attn_w_normalized
    

class MH_SelfAttention(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads):
        
        super().__init__()
        
        layers = [SH_SelfAttention(input_size) for i in range(num_attn_heads)]
        
        self.multihead_pipeline = nn.ModuleList(layers)
        embed_size = input_size
        self.Wz = nn.Linear(num_attn_heads*embed_size, embed_size)
        
    
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, deepadr similarity type vector, input_size)
        """
        
        out = []
        for SH_layer in self.multihead_pipeline:
            z, __ = SH_layer(X)
            out.append(z)
        # concat on the feature dimension
        out = torch.cat(out, -1) 
        
        # return a unified vector mapping of the different self-attention blocks
        return self.Wz(out)
        

class TransformerUnit(nn.Module):
    
    def __init__(self, input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout):
        
        super().__init__()
        
        embed_size = input_size
        self.multihead_attn = MH_SelfAttention(input_size, num_attn_heads)
        
        self.layernorm_1 = nn.LayerNorm(embed_size)
        
        self.MLP = nn.Sequential(
            nn.Linear(embed_size, embed_size*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_size*mlp_embed_factor, embed_size)
        )
        
        self.layernorm_2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(p=pdropout)
                
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, deepadr similarity type vector, input_size)
        """
        # z is tensor of size (batch, deepadr similarity type vector, input_size)
        z = self.multihead_attn(X)
        # layer norm with residual connection
        z = self.layernorm_1(z + X)
        z = self.dropout(z)
        z_ff= self.MLP(z)
        z = self.layernorm_2(z_ff + z)
        z = self.dropout(z)
        
        return z
        
class FeatureEmbAttention(nn.Module):
    def __init__(self, input_dim):
        '''
        Args:
            input_dim: int, size of the input vector (i.e. feature vector)
        '''

        super().__init__()
        self.input_dim = input_dim
        # use this as query vector against the transformer outputs
        self.queryv = nn.Parameter(torch.randn(input_dim, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1) # normalized across seqlen

    def forward(self, X):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (batch, ddi similarity type vector, feat_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4))
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_weights = X_scaled.matmul(queryv_scaled)
        
        # softmax
        attn_weights_norm = self.softmax(attn_weights)

        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, num similarity type vectors)
        # perform batch multiplication with X that has shape (bsize, num similarity type vectors, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_weights_norm.unsqueeze(1).bmm(X).squeeze(1)
        
        # returns (bsize, feat_dim), (bsize, num similarity type vectors)
        return z, attn_weights_norm
    
class FeatureEmbDrugAttention(nn.Module):
    def __init__(self, input_dim, X_dim):
        '''
        Args:
            input_dim: int, size of the input vector (i.e. feature vector)
        '''

        super().__init__()
        self.input_dim = input_dim
        self.X_dim = X_dim
        # use this as query vector against the transformer outputs
        self.queryv = nn.Parameter(torch.randn(input_dim, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1) # normalized across seqlen

    def forward(self, X):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (batch, ddi similarity type vector, feat_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4))
        
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_weights = X_scaled.matmul(queryv_scaled)

        # softmax
        attn_weights_norm = self.softmax(attn_weights)[:, :self.X_dim]

        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, num similarity type vectors)
        # perform batch multiplication with X that has shape (bsize, num similarity type vectors, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_weights_norm.unsqueeze(1).bmm(X[:, :self.X_dim, :]).squeeze(1)
        
        # returns (bsize, feat_dim), (bsize, num similarity type vectors)
        return z, attn_weights_norm
    
class GeneEmbAttention(nn.Module):
    def __init__(self, input_dim):
        '''
        Args:
            input_dim: int, size of the input vector (i.e. feature vector)
        '''

        super().__init__()
        self.input_dim = input_dim
        # use this as query vector against the transformer outputs
#         self.queryv = nn.Parameter(torch.randn((input_dim,input_dim), dtype=torch.float32), requires_grad=True)
        self.queryv = nn.Parameter(torch.randn((1,1), dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1) # normalized across seqlen

    def forward(self, X):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (batch, deepadr similarity type vector, feat_dim), dtype=torch.float32
        '''
        
        X = X.squeeze(1).unsqueeze(2)

        X_scaled = X / (self.input_dim ** (1/4))
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_weights = X_scaled.mul(queryv_scaled).squeeze(1)

        
        # softmax
        attn_weights_norm = self.softmax(attn_weights)

        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, num similarity type vectors)
        # perform batch multiplication with X that has shape (bsize, num similarity type vectors, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = X.squeeze(1).mul(attn_weights_norm)
        
        z_ret = z.squeeze(2)
        attn_ret = attn_weights_norm.squeeze(2)
        
        # returns (bsize, feat_dim), (bsize, num similarity type vectors)
        return z_ret, attn_ret



#CNN
class _ResNet_Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, prev_layer_depth, expansion=4, stride_3x3=1,  padding_3x3=1, conv_identity=False, stride_conv_identity=1, PReLU=True):
        super(_ResNet_Bottleneck, self).__init__()
        """ ResNet Bottleneck
            :inplanes: int, number of input channels
            :planes: int, number of output channels
            :prev_layer_depth: int, number of previous layers
            :expansion: int, expansion factor
            :stride_3x3: int, stride of 3x3 convolution
            :padding_3x3: int, padding of 3x3 convolution
            :conv_identity: bool, whether to use identity mapping
            :stride_conv_identity: int, stride of identity mapping
            :PReLU: bool, whether to use PReLU activation
        """

        self.outplanes = planes*expansion
        self.conv_identity = conv_identity
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(prev_layer_depth)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride_3x3, padding=padding_3x3)
        self.conv2_bn = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=expansion*planes, kernel_size=1)
        self.conv3_bn = nn.BatchNorm2d(planes)
        if conv_identity:
            self.conv_identity_layer = nn.Conv2d(inplanes, planes*expansion, kernel_size=1, stride=stride_conv_identity)
        self._PReLU = PReLU
        if self._PReLU:
            self.conv1_PReLU = nn.PReLU()
            self.conv2_PReLU = nn.PReLU()
            self.conv3_PReLU = nn.PReLU()

    def forward(self, x, activation=F.relu):
        """ Forward pass
            :x: torch.Tensor, input features
        """
        identity = x
        if self._PReLU:
            out = self.conv1(self.conv1_PReLU(self.conv1_bn(x)))
            out = self.conv2(self.conv2_PReLU(self.conv2_bn(out)))
            out = self.conv3(self.conv3_PReLU(self.conv3_bn(out)))
        else:
            out = self.conv1(activation(self.conv1_bn(x)))
            out = self.conv2(activation(self.conv2_bn(out)))
            out = self.conv3(activation(self.conv3_bn(out)))
        if self.conv_identity:
            identity = self.conv_identity_layer(x)
        out += identity
        return out

class ResNet(nn.Module):
    def __init__(
            self,
    ) -> None:
        super(ResNet, self).__init__()
        """ ResNet constructor
            CNN tower network
        """

        # define stem
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(16)

        # define stages
        self.stage_1a = self._make_layer(inplanes=16, planes=16, prev_layer_depth=16, blocks=1, conv_identity=True)
        self.stage_1b = self._make_layer(inplanes=64, planes=16, prev_layer_depth=64, blocks=2)
        self.stage_2a = self._make_layer(inplanes=64, planes=32, prev_layer_depth=64, blocks=1)
        self.stage_2b = self._make_layer(inplanes=128, planes=32, prev_layer_depth=128, blocks=3)
        self.stage_3a = self._make_layer(inplanes=128, planes=64, prev_layer_depth=128, blocks=1)
        self.stage_3b = self._make_layer(inplanes=256, planes=64, prev_layer_depth=256, blocks=5)
        self.stage_4a = self._make_layer(inplanes=256, planes=128, prev_layer_depth=256, blocks=1)
        self.stage_4b = self._make_layer(inplanes=512, planes=128, prev_layer_depth=512, blocks=2)

        # define output transformation
        self.conv5a_bn = nn.BatchNorm2d(512)
        self.conv5a = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        self.conv5b_bn = nn.BatchNorm2d(128)
        self.conv5b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv5c_bn = nn.BatchNorm2d(128)
        self.conv5c = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1)

        self.drop = nn.Dropout(p=0.2)
        self.avgPool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, input, activation=F.relu):
        """ Forward pass
            :input: torch.Tensor, input features
            :activation: torch.nn.functional, activation function
         """
        output = F.relu(self.conv1_bn(self.conv1(input)))
        output = self.stage_1a(output)
        output = self.stage_1b(output)
        output = self.avgPool(output)
        output = self.stage_2a(output)
        output = self.stage_2b(output)
        output = self.avgPool(output)
        output = self.stage_3a(output)
        output = self.stage_3b(output)
        output = self.avgPool(output)
        output = self.stage_4a(output)
        output = self.stage_4b(output)

        output = F.relu(self.conv5a(self.conv5a_bn(output)))
        output = F.relu(self.conv5b(self.conv5b_bn(output)))
        output = F.relu(self.conv5c(self.conv5c_bn(output)))

        return output

    def _make_layer(self, inplanes, planes, blocks, prev_layer_depth, expansion=4, stride_3x3=1, padding_3x3=1,
                    conv_identity=True, stride_conv_identitiy=1, PReLU=True):
        """ Helper function, make a layer of ResNet
           :inplanes: int, input channels
           :planes: int, output channels
           :blocks: int, number of blocks
           :prev_layer_depth: int, previous layer depth
           :expansion: int, expansion factor
           :stride_3x3: int, stride of 3x3 convolution
           :padding_3x3: int, padding of 3x3 convolution
           :conv_identity: bool, whether to use identity mapping
           :stride_conv_identity: int, stride of identity mapping
           :PReLU: bool, whether to use PReLU activation
        """
        layers = []
        for _ in range(blocks):
            layers.append(_ResNet_Bottleneck(inplanes, planes, prev_layer_depth=prev_layer_depth, stride_3x3=stride_3x3,
                                             padding_3x3=padding_3x3, conv_identity=conv_identity,
                                             stride_conv_identity=stride_conv_identitiy, PReLU=PReLU))

        return nn.Sequential(*layers)

# 这是消融实验中的简单MLP
class SimpleMLP(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(908, 512)  # 从输入层到隐藏层1
        self.fc2 = nn.Linear(512, 512)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(512, 496)  # 隐藏层2到输出层
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # 应用Dropout
        x = self.fc3(x)
        # 注意：通常不在输出层之前应用Dropout
        return x


class DDCL_CNN(nn.Module):
    def __init__(self,):
        super(DDCL_CNN, self).__init__()
        """ CNN tower constructor
        """
        self.avg_pool2d = nn.AvgPool2d(kernel_size=4,stride=4)
        self.cnn = ResNet()
        self.cnn_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 510),
            nn.PReLU(),
            nn.BatchNorm1d(510),
            nn.Linear(in_features=510, out_features=500),
            nn.PReLU(),
            nn.Linear(in_features=500, out_features=496)
        )

    def forward(self, imgs):
        """ Forward pass
           :imgs: torch.Tensor, image features
        """
        # imgs = torch.sum(imgs,dim=1)
        output = self.avg_pool2d(imgs)
        # print(output.shape)
        output = output.reshape(-1,1,16,16)
        output = self.cnn_projection(self.cnn(output))
        return output
