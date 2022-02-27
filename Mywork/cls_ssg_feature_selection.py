# -*- coding: utf-8 -*-
# +
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
BASE_DIR =  '/mnt/workspace/Github/Pointnet_Pointnet2_pytorch'
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + '/models')
from pointnet2_utils import PointNetSetAbstraction

from copy import deepcopy
from pprint import pprint as pprint


# +
def print_proc_time(f):
    """ 計測デコレーター """
    def print_proc_time_func(*args, **kwargs):
        start_time = time.perf_counter()

        return_val = f(*args, **kwargs)

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print(f.__name__, elapsed_time)
        
        return return_val
    
    return print_proc_time_func

# @print_proc_time
# def calc_mass(n, m):
#     """"" 計測対象関数 """""
#     for i in range(n):
#         x = i ** m
        
# calc_mass(3000000, 3)


# +
# @print_proc_time
# def to_zero(feature, idx):
# #     input [B, 1024]
# #     idx list
# #     output [B, 1024]
#     processed_feature = deepcopy(feature)
#     for i in idx:
#         processed_feature[:, i] = 0
    
#     return processed_feature

# @print_proc_time
def to_zero(feature, idx):
#     input [B, 1024] torchかnumpy
#     idx list
#     output [B, 1024]
    idx = list(idx)
    processed_feature = deepcopy(feature)
    processed_feature[:, idx] = 0
    
    return processed_feature


# +
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, idx):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)
        
        x_processed = l3_points.view(B, 1024)
        x_processed = to_zero(x_processed, idx) #これを挿入
        x_processed = self.drop1(F.relu(self.bn1(self.fc1(x_processed))))
        x_processed = self.drop2(F.relu(self.bn2(self.fc2(x_processed))))
        x_processed = self.fc3(x_processed)
        x_processed = F.log_softmax(x_processed, -1)
        
        

#         return x, l3_points
        return x_processed, l3_points


# +
class get_first_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_first_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)
        
        l3_points = l3_points.view(B, 1024)
#         x_processed = to_zero(x_processed, idx) #これを挿入
#         x_processed = self.drop1(F.relu(self.bn1(self.fc1(x_processed))))
#         x_processed = self.drop2(F.relu(self.bn2(self.fc2(x_processed))))
#         x_processed = self.fc3(x_processed)
#         x_processed = F.log_softmax(x_processed, -1)
        
        

#         return x, l3_points
        return l3_points


# +
class get_second_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_second_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, l3_points, idx=[]):
        B, _ = l3_points.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)
        
        x_processed = l3_points.view(B, 1024)
        x_processed = to_zero(x_processed, idx) #これを挿入
        x_processed = self.drop1(F.relu(self.bn1(self.fc1(x_processed))))
        x_processed = self.drop2(F.relu(self.bn2(self.fc2(x_processed))))
        x_processed = self.fc3(x_processed)
        x_processed = F.log_softmax(x_processed, -1)
        
        

#         return x, l3_points
        return x_processed, l3_points


# -

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

# +
# import torch

# size = 10000
# batch_size = 10
# a = torch.arange(size)
# a = torch.randperm(size)[:256]
# print("a:", a)

# f = torch.randint(0, 10, (batch_size, size ))
# print("f:", f)

# f1 = to_zero(f, a)
# print("f1: ",f1)

# f2 = to_zero2(f, a)
# print("f2: ",f2)


# +
# import time

# start_time = time.perf_counter()

# time.sleep(1)

# end_time = time.perf_counter()

# elapsed_time = end_time - start_time

# print(elapsed_time)

# +
# import time

# start_time = time.process_time()

# time.sleep(1)

# end_time = time.process_time()

# elapsed_time = end_time - start_time

# print(elapsed_time)
