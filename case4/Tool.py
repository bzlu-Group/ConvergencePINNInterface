# -*- coding: utf-8 -*-

import numpy as np 
import torch
import torch.nn as nn
import random

def r_boundary(theta):

    return 1.5*(1.5+0.14*torch.sin(4*theta)+0.12*torch.cos(6*theta)+0.09*torch.cos(5*theta))

def r_interface(theta):

    return 2*(0.5+0.18*torch.sin(3*theta)+0.08*torch.cos(2*theta)+0.2*torch.cos(5*theta))


def data_transform_theta_interface(theta,device): 
    x=r_interface(theta)*torch.cos(theta)
    y=r_interface(theta)*torch.sin(theta)   
    input=torch.cat((x,y),dim=1) 

    return x,y,input

def data_transform_theta_boundary(psi,device): 
    x=r_boundary(psi)*torch.cos(psi)
    y=r_boundary(psi)*torch.sin(psi)   
    input=torch.cat((x,y),dim=1) 

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
