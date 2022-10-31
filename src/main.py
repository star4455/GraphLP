import torch
import numpy as np
import time
import argparse
import os.path
from model import LinkPred
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import prepare_data, draw, draw1, edge_index_to_matrix, matrix_to_scores_labels
import torch.nn.functional as F
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='Link Prediction based on Collaborative Pooling Graph Convolutional Networks')
'''Dataset'''
parser.add_argument('--data-name', default='NS', help='graph name')
parser.add_argument('--seed', type=int, default=1,  help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.10,
                    help='ratio of test links')
parser.add_argument('--val-ratio', type=float, default=0.05,
                    help='ratio of validation links.')
parser.add_argument('--perturbation-ratio', type=float, default=0.1,
                    help='ratio of graph structure perturbation.')
parser.add_argument('--perturbation-num', type=int, default=12,
                    help='total number of graph structure perturbation.')

'''Model and Training'''
parser.add_argument('--lr', type=float, default=0.0012, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lamda', type=float, default=0.13)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--epoch-num', type=int, default=200)
parser.add_argument('--MSE', type=str, default=False)
parser.add_argument('--log', type=str, default=None, help='log by tensorboard, default is None')

args = parser.parse_args()

print ("-"*42+'Dataset and Features'+"-"*43)
print ("{:<10}|{:<10}|{:<10}|{:<10}|{:<10}"\
    .format('Dataset','Perturbation Size','Test Ratio','Val Ratio','Perturbation Ratio'))
print ("-"*105)
print ("{:<10}|{:<18}|{:<10}|{:<10}|{:<10}"\
    .format(args.data_name,args.perturbation_num,args.test_ratio,args.val_ratio,args.perturbation_ratio))
print ("-"*105)


print('<<Begin generating training data>>')

train_loader, val_loader, test_loader, true_graph = prepare_data(args)

print('<<Complete generating training data>>')


print ("-"*42+'Model and Training'+"-"*45)
print ("{:<13}|{:<13}|{:<13}|{:<15}|{:<13}|{:<13}"\
    .format('Learning Rate','Batch Size','Epoch','Weight-decay','Dropout','Lamda'))
print ("-"*105)

print ("{:<13}|{:<13}|{:<13}|{:<13}|{:<13}|{:<13}"\
    .format(args.lr, args.batch_size, args.epoch_num, args.weight_decay, args.dropout, args.lamda))
print ("-"*105)




#获得图邻接矩阵上三角的信息
data_matrix = edge_index_to_matrix(true_graph)
true_labels = matrix_to_scores_labels(data_matrix)
true_labels = true_labels.to(device)
#Link prediction模型初始化
hidden_channels = data_matrix.size()[0]
print("hidden_channels are:", hidden_channels)
model = LinkPred(hidden_channels, args.dropout, args.lamda).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss(reduction='mean')


def train(loader):
    model.train()#启用 BatchNormalization 和 Dropout, 让model变成训练模式
    loss_epoch = 0
    for data in tqdm(loader, desc="train"):  # Iterate in batches over the training dataset.
        # print("the type of data:", data.deletion)
        data = data.to(device)#表示将构建的张量或者模型分配到相应的设备上
        out = model(data)
        torch.cuda.empty_cache()#显存释放机制
        out = out.to(device)
        # loss = criterion(out.view(-1).float(), true_labels.float())#view(-1)将原张量变成一维结构；true_graph：张量形式的边信息
        loss = F.binary_cross_entropy(out.view(-1).float(), true_labels.float())
        optimizer.zero_grad()#梯度归零
        loss = loss.to(torch.float32)
        loss.backward()
        #更新所有的参数（mini-batch训练模式是假定每一个训练集只有mini-batch这样大，将每一次mini-batch看做是一次训练，一次训练更新一次参数）
        optimizer.step()
        loss_epoch = loss_epoch + loss.item()
    return loss_epoch/len(loader), out



def test(loader,data_type='val'):
    '''
    model.eval()，不启用 BatchNormalization 和 Dropout;
    model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变;
    对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
    '''
    model.eval()
    scores = torch.tensor([])
    labels = torch.tensor([]).to(torch.float32)
    loss_total=0
    with torch.no_grad():
        for data in tqdm(loader,desc='test:'+data_type):  # Iterate in batches over the test dataset.
            data = data.to(device)
            out = model(data)
            loss = F.binary_cross_entropy(out.view(-1).float(), true_labels.float())
            loss_total=loss_total+loss.item()
            out = out.cpu().clone().detach()
            ####################以下用于全网络训练#############################
            scores = torch.cat((scores,out),dim = 0) # predicted scores
            labels = torch.cat((labels,true_labels.cpu().clone().detach().float()),dim = 0) # true labels
            scores = scores.cpu().clone().detach().numpy()
            labels = labels.cpu().clone().detach().numpy()

            return roc_auc_score(labels, scores), average_precision_score(labels, scores),loss_total



Best_Val_fromloss=1e10
Final_Test_AUC_fromloss=0
Final_Test_AP_fromloss=0

Best_Val_fromAUC=0
Final_Test_AUC_fromAUC=0
Final_Test_AP_fromAUC=0
Best_Precision = 0

history = defaultdict(list)
for epoch in range(0, args.epoch_num):
    #模型训练, out 为训练后的相似性矩阵
    loss_epoch, out = train(train_loader)
    history['loss_epoch'].append(loss_epoch)
    #模型验证
    val_auc, val_ap, val_loss = test(val_loader, data_type='val')
    # print(f'val_auc:{val_auc:.4f}, val_ap:{val_ap:.4f},val_loss are:{val_loss:0.4f}')
    history['val_auc'].append(val_auc)
    history['val_ap'].append(val_ap)
    history['val_loss'].append(val_loss)
    #模型测试
    test_auc, test_ap, _ = test(test_loader, data_type='test')
    if val_loss < Best_Val_fromloss:
        Best_Val_fromloss = val_loss
        Final_Test_AUC_fromloss = test_auc
        Final_Test_AP_fromloss = test_ap

    if val_auc > Best_Val_fromAUC:
        Best_Val_fromAUC = val_auc
        Final_Test_AUC_fromAUC = test_auc
        Final_Test_AP_fromAUC = test_ap

    print(f'Epoch: {epoch:03d}, Loss : {loss_epoch:.4f}, Val Loss : {val_loss:.4f}, \
           Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, Test AP: {Final_Test_AP_fromAUC:.4f}, Picked AUC:{Final_Test_AUC_fromAUC:.4f}')

draw(history)
draw1(history)
print(f'From loss: Final Test AUC: {Final_Test_AUC_fromloss:.4f}, Final Test AP: {Final_Test_AP_fromloss:.4f}')
print(f'From AUC: Final Test AUC: {Final_Test_AUC_fromAUC:.4f}, Final Test AP: {Final_Test_AP_fromAUC:.4f}')
