# -*- coding: utf-8 -*-
import torch
import numpy as np

pi=float(np.pi)
torch.set_default_dtype(torch.float32)

class Data(object):
    def __init__(self,r0,L,box,device):
        self.r0=r0
        self.L=torch.tensor(L).to(device)
        self.box=box
        self.device=device

    def SampleFromGamma(self,num):
        """
        L : the center of sphere
        output: boundary point and related f_direction 
        """
        L=self.L
        psi=2*pi*torch.rand(num,device=self.device).view(-1,1)   #[0,2pi]
        x=L[0]+self.r0*torch.cos(psi)
        y=L[1]+self.r0*torch.sin(psi)   
        X=torch.cat((x,y),dim=1) 
           
        f_direction=(X-self.L)/self.r0
        
        return psi.to(self.device),f_direction.to(self.device)
    

    def SampleFromDomain(self,num):
        """
        num: the number of training points
        """
        L=self.L
        X=self.__sampleFromDomain(num)      
        y=torch.norm(X-L,dim=1)     
        location=torch.where(y>self.r0)[0]
        X_out=X[location,:]      
        
        location=torch.where(y<self.r0)[0]
        X_in=X[location,:]      

        return X_out,X_in


    def __sampleFromDomain(self,num):
        xmin,xmax,ymin,ymax=self.box
        x = torch.rand(10*num,device=self.device).view(-1,1) * (xmax - xmin) + xmin
        y = torch.rand(10*num,device=self.device).view(-1,1) * (ymax - ymin) + ymin
        X = torch.cat((x,y),dim=1)
        y=torch.norm(X-self.L,dim=1)     
        location=torch.where(y<self.box[1])[0]
        X=X[location,:] 
        X=X[:num]

        return X


    def SampleFromBoundary(self,num):

        L=self.L
        psi=2*pi*torch.rand(num,device=self.device).view(-1,1)   #[0,2pi]
        return psi.to(self.device)

