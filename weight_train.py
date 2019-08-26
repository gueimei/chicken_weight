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

## data loader and set
def load_xml(): #load every xml file
    sets = [('weight_train_name')]
    ac_data = []
    train_data = []
    #classes = ['chicken']
    for files in sets:
        if not os.path.exists('D:\\lab\\chicken_video\\dataset\\weight_test'):
            os.makedirs('D:\\lab\\chicken_video\\dataset\\weight_test')
    xml_ids = open('D:\\lab\\chicken_video\\dataset\\%s.txt'%(files)).read().strip().split()
    
    for ID in xml_ids:
        xml_source = "D:\\lab\\chicken_video\\dataset\\weight_xml\\%s.xml"%(ID)
        load_Data(xml_source, ac_data, train_data)
    
    return ac_data, train_data

        
def load_Data(source, ac_data, train_data): ##load necessary data from a xml file(object coordinate)
    xml = open(source)
    tree=ET.parse(xml)
    root = tree.getroot()
    times = 0
    for obj in root.iter('object'):
        ac_data.append(float(obj[0].text)) ##load ac_data
        times += 1
        if times % 1 == 0:
            xmlbox = obj.find('bndbox')
            ##box -> {[0]==x,[1]==y,[2]==x_max,[3]==y_max}
            box = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            train_data.append(calculate_Chicken_Area(box[0], box[1], box[2], box[3])) ##load train_data
        
def calculate_Chicken_Area(xmin, ymin, xmax, ymax):
    return float((xmax-xmin) * (ymax-ymin))

## build net
def build_Net():
    net = nn.Sequential(
        nn.Linear(1, 10), #(input nerual, output nerual)
        nn.ReLU(),
        nn.Linear(10, 1)
        )
    return net


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
    LR = 0.2
    MOMENTUM = 0.8
    
    ##load data
    ac_data, train_data = load_xml()
    ##list->tensor
    train_data = torch.unsqueeze(torch.tensor(train_data), dim=1)
    ac_data = torch.unsqueeze(torch.tensor(ac_data), dim=1)
    
    
    dataset = Data.TensorDataset(train_data, ac_data) ##(train data, target data)
    data_loader = Data.DataLoader(
            dataset = dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 2)
    
    net = build_Net()
    
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum = MOMENTUM)
    loss_func = torch.nn.MSELoss()
    
    #train
    for i in range (EPOCH):
        for step, (batch_train, batch_ac) in enumerate(data_loader):
            print('Epoch: ', i, '| Step: ', step)
        
            prediction = net(batch_train)
            loss = loss_func(prediction, batch_ac)
            print("loss:", loss.data)
        
            opt.zero_grad()   # clear gradients for next train
            loss.backward()   # backpropagation, compute gradients
            opt.step()
        
        if i % 9 == 0:
            torch.save(net.state_dict(), 'chicken_weight_%d.pth',i)
        