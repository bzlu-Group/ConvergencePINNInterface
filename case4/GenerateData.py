# -*- coding: utf-8 -*-
import torch
import numpy as np

pi=float(np.pi)
torch.set_default_dtype(torch.float32)

def r_boundary(theta):

    return 1.5*(1.5+0.14*torch.sin(4*theta)+0.12*torch.cos(6*theta)+0.09*torch.cos(5*theta))

def r_interface(theta):

    return 2*(0.5+0.18*torch.sin(3*theta)+0.08*torch.cos(2*theta)+0.2*torch.cos(5*theta))

def f_interface(theta):
    aa=(0.54*torch.cos(3*theta)-0.16*torch.sin(2*theta)-torch.sin(5*theta))*torch.cos(theta)
    x=2*aa-r_interface(theta)*torch.sin(theta) #\farc{\partial x}{\partial \theta}
    y=2*aa+r_interface(theta)*torch.cos(theta) #\farc{\partial y}{\partial \theta}
    X=torch.cat((y,-x),dim=1)   
    return X


class Data_Single(object):
    def __init__(self,device):
        self.device=device

    def sphere_bound(self,num):
        """
        L : the center of sphere
        output: boundary point and related f_direction 
        """
        theta=2*pi*torch.rand(num,device=self.device).view(-1,1)   #[0,2pi]
        x=r_interface(theta)*torch.cos(theta)
        y=r_interface(theta)*torch.sin(theta)   
        X=torch.cat((x,y),dim=1) 
    
        f_direction=f_interface(theta)
        
        return theta.to(self.device),f_direction.to(self.device)

    def sampleDomain_hole(self,num):
        xmin,xmax,ymin,ymax=-3,3,-3,3
        x = torch.rand(4*num).view(-1,1) * (xmax - xmin) + xmin
        y = torch.rand(4*num).view(-1,1) * (ymax - ymin) + ymin
        X = torch.cat((x,y),dim=1)

        rr=torch.tensor(torch.sqrt((x**2+y**2)))      
        theta=-1000*torch.ones_like(x)
        # I
        index_x=torch.where(x>=0)[0]
        index_y=torch.where(y>=0)[0]
        loca=list(set(index_x.numpy())&set(index_y.numpy()))
        theta[loca]=torch.arccos(x[loca]/rr[loca])
        # II
        index_x=torch.where(x<0)[0]
        index_y=torch.where(y>0)[0]
        # print(index_x)
        loca=list(set(index_x.numpy())&set(index_y.numpy()))
        theta[loca]=pi-torch.arccos(-x[loca]/rr[loca]) 
        # III
        index_x=torch.where(x<0)[0]
        index_y=torch.where(y<0)[0]
        loca=list(set(index_x.numpy())&set(index_y.numpy()))
        theta[loca]=pi+torch.arccos(-x[loca]/rr[loca])   
        # IIII
        index_x=torch.where(x>0)[0]
        index_y=torch.where(y<0)[0]
        loca=list(set(index_x.numpy())&set(index_y.numpy()))
        theta[loca]=2*pi-torch.arccos(x[loca]/rr[loca])      
        

        r_b=r_boundary(theta)  
        location=torch.where(rr<r_b)[0]
        X=(X[location,:])[:num]
        rr=(rr[location,:])[:num]
        theta=(theta[location,:])[:num]

        r_in=r_interface(theta)
        location=torch.where(rr<r_in)[0]
        X_in=(X[location,:])
        location=torch.where(rr>=r_in)[0]
        X_out=(X[location,:])

        return X_out.to(self.device),X_in.to(self.device)


    def sampleFromBoundary(self,num):

        psi=2*pi*torch.rand(num,device=self.device).view(-1,1)   #[0,2pi]

        return psi.to(self.device)

