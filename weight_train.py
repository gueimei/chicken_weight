# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 00:58:03 2019

@author: may
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import argparse
import os
import xml.etree.ElementTree as ET
from pre_train import load_xml
import numpy as np



## build net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(8, 16)
        self.hidden1 = nn.Linear(16, 8)
        self.hidden2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)
    def forward(self, data):
        pred = nn.functional.relu(self.input(data))
        pred = nn.functional.relu(self.hidden1(pred))
        pred = nn.functional.relu(self.hidden2(pred))
        out = self.output(pred)
        return out


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",dest='epoch', help = "number of epochs" 
                        ,default = 100, type = int)
    parser.add_argument("--batch_size",dest='batch_size', help = "size of each data batch" 
                        ,default = 10, type = int)

if __name__ == "__main__":
    args = arg_parse()
    EPOCH = 100 #args.epoch
    BATCH_SIZE = 10 #args.batch_size
    LR = 0.005
    MOMENTUM = 0.8
    
    ##load data
    ac_data, train_data = load_xml()
    ##list->tensor
    train_data = torch.Tensor(train_data)
    ac_data = torch.unsqueeze(torch.tensor(ac_data), dim=1).cuda()
    
    
    dataset = Data.TensorDataset(train_data, ac_data) ##(train data, target data)
    data_loader = Data.DataLoader(
            dataset = dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 0)
    
    net = Net()
    if(torch.cuda.is_available()):
        net.cuda()
    
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum = MOMENTUM)
    
    
    #train
    for i in range (EPOCH):
        for step, (batch_train, batch_ac) in enumerate(data_loader):
            batch_train = torch.autograd.Variable(batch_train).cuda()
            batch_ac = torch.autograd.Variable(batch_ac).cuda()
            print('Epoch: ', i, '| Step: ', step)

            prediction = net(batch_train)
            loss = loss_func(prediction, batch_ac)
            #print(prediction)
            print("loss:", loss.data)
        
            opt.zero_grad()   # clear gradients for next train
            loss.backward()   # backpropagation, compute gradients
            opt.step()
        
        if i % 9 == 0 and i != 0:
            torch.save(net.state_dict(), "chicken_weight_%s.pth" %(i))
        
        