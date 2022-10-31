import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# from torch_geometric.nn import GCNConv
# import torch_geometric.nn as gnn
from utils import edge_index_to_matrix, process

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinkPred(nn.Module):
    def __init__(self, nhid, dropout, lamda, num_inputs = 5):
        super(LinkPred, self).__init__()
        self.lo1 = LO(nhid, nhid) #X_1 = A*Z
        self.gc1 = GraphConvolution(nhid, nhid)#A*X_1*W
        self.lo2 = LO(nhid, nhid) #X_2 = A*Z
        self.gc2 = GraphConvolution(nhid, nhid)#A*X_2*W
        self.lo3 = LO(nhid, nhid) #X_3 = A*Z
        self.mlp = MLP(nhid*nhid, num_inputs)
        self.dropout = dropout
        self.lamda = lamda

    def forward(self, graph):
        adj = edge_index_to_matrix(graph)
        normalize = torch.from_numpy(np.identity(adj.size()[0]))
        # print("the adj is:", adj)
        #对数据进行预处理，x= D-1/2*(A+I)*D-1/2
        normalize = torch.from_numpy(process(normalize.numpy(), adj.numpy()))
        #以邻接矩阵的形式输入
        adj = adj.to(device)
        adj = adj.to(torch.float32)
        adj = F.dropout(adj, self.dropout, training=self.training)
        ##第一层
        adj = self.lo1(adj, self.lamda)
        x1 = adj
        #GCN2 + LO2
        adj = self.gc1(adj, normalize)
        x2 = adj
        ###第二层
        adj = self.lo2(adj, self.lamda)
        x3 = adj
        # GCN3 + LO3
        adj = self.gc2(adj, normalize)
        x4 = adj

        ###第三层
        adj = self.lo3(adj, self.lamda)
        x5 = adj


         #Classifier

        feature1 = torch.flatten(x1, start_dim=0) #torch.from_numpy(x1.data_matrix.flatten())
        feature2 = torch.flatten(x2, start_dim=0)#torch.from_numpy(x2.data_matrix.flatten())
        feature3 = torch.flatten(x3, start_dim=0)#torch.from_numpy(x.data_matrix.flatten())
        feature4 = torch.flatten(x4, start_dim=0)#torch.from_numpy(x.data_matrix.flatten())
        feature5 = torch.flatten(x5, start_dim=0)#torch.from_numpy(x.data_matrix.flatten())
        # feature6 = torch.flatten(x6, start_dim=0)#torch.from_numpy(x.data_matrix.flatten())

        feature1 = feature1.reshape(1, len(feature1))
        feature2 = feature2.reshape(1, len(feature2))
        feature3 = feature3.reshape(1, len(feature3))
        feature4 = feature4.reshape(1, len(feature4))
        feature5 = feature5.reshape(1, len(feature5))

        #计算3层
        features = torch.cat((feature1, feature2, feature3, feature4, feature5),0).t()
        out = self.mlp(features)
        return out


class LO(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(LO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, adjacency, lamda):
        adjacency = adjacency.to(device)
        m, n = self.in_features, self.out_features  #adjacency.shape
        X1 = torch.tensor(lamda)*adjacency
        X2 = torch.tensor(lamda)*torch.mm(adjacency.T, adjacency)
        X1 = X1.to(device)
        X2 = X2.to(device)
        output = torch.mm(X1,((X2 + torch.eye(m).to(device))).inverse()).mm(adjacency.T).mm(adjacency)


        return output


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, adj, normal): #input:adj（输入的数据）, adj:x(进行度归一化后的矩阵)
        adj = adj.to(device)
        normal = normal.to(device)
        adj = adj.to(torch.float32)
        normal = normal.to(torch.float32)
        support = torch.mm(adj, self.weight)
        output = torch.spmm(normal, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MLP(torch.nn.Module):
    # adopt a MLP as aggregator for graphs
    def __init__(self, entries_size, input_size):
        super(MLP, self).__init__()
        self.nn = nn.BatchNorm1d(input_size)
        self.linear1 = torch.nn.Linear(input_size,input_size)
        self.linear2 = torch.nn.Linear(input_size,1)
    def forward(self, features):
        out= self.nn(features)
        out= self.linear1(out)
        out= self.linear2(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = torch.sigmoid(out)
        return out
