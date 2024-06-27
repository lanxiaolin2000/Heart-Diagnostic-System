# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:49:43 2024

@author: Xiaolin
"""
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义模型的层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=21, padding=10)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=21, padding=10)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=21, padding=10)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.batch_norm1(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.batch_norm2(x)
        return x