# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:11:43 2019

@author: may
"""

from pre_train import collect_data
import math

def get_numeralData(inp, img):
    numeral_list = []
    for i in range(inp.shape[0]):
       data  = collect_data(inp[i,0], inp[i,1], inp[i,2], inp[i,3], img)
       
       if data == 0:
            numeral_list.pop()
            print("pop")
       else:
            data = list(map(lambda x: math.log(x), data))
            numeral_list.append(data)
    
    print(numeral_list)