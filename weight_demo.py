# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:11:43 2019

@author: may
"""

from pre_train import collect_data
import math
import torch.nn as nn
import torch

def get_numeralData(inp, img):
    numeral_list = []
    for i in range(inp.shape[0]):
       data  = collect_data(int(inp[i,0]), int(inp[i,1]), math.ceil(inp[i,2]), math.ceil(inp[i,3]), img)
       
       if data[6] == 0:
            print("pop")
       else:
            data = list(map(lambda x: math.log(x), data))
       numeral_list.append(data)
     
    return numeral_list
       
       
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

def predict_weight(prediction_data):
    weight_path = "/workspace/weight_dataset/chicken_weight_first.pth"
    model_net = Net()
    model_net.load_state_dict(torch.load(weight_path)) #load checkpoint
    if(torch.cuda.is_available()):
        model_net.cuda()
        
    model_net.eval()  # Set in evaluation mode
    
    prediction_data = torch.Tensor(prediction_data)
    if(torch.cuda.is_available()):
        prediction_data = prediction_data.cuda()
    with torch.no_grad(): 
        prediction = model_net(prediction_data)
        
    return prediction
    