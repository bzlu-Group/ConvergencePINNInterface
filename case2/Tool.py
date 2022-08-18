# -*- coding: utf-8 -*-

import numpy as np 
import torch
import torch.nn as nn
import random

def data_transform_theta(psi,L,r0,device): 
    x=L[0]+r0*torch.cos(psi)
    y=L[1]+r0*torch.sin(psi)   
    input=torch.cat((x,y),1)

    return x,y,input

def data_transform(data,device): 
    x=data[0].view(-1,1)
    y=data[1].view(-1,1)
    x=x.clone().detach().requires_grad_(True)
    y=y.clone().detach().requires_grad_(True)
    input=torch.cat((x,y),1)

    return x,y,input


def gradient(data,x,y,device):
    dx=grad(data,x,device)
    dy=grad(data,y,device)

    return dx,dy


def grad(y,x,device):
    '''return tensor([dfdx,dfdy,dfdz])
    '''    
    dydx, = torch.autograd.grad(outputs=y,inputs=x,retain_graph=True,grad_outputs=torch.ones(y.size()).to(device) ,
                                create_graph=True,allow_unused=True)
    return dydx
