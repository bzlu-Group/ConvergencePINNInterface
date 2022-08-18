import numpy as np 
import argparse
import torch
import time,os
import itertools
import random
import torch.optim as optim
from Tool import grad, data_transform, gradient,data_transform_theta
from Net_type import FNN
from GenerateData import Data_Single

############ exact solution ###############
def u(x,label,a):

    x=x.t()  
    if label=='inner':
        u=(a*torch.tanh(x[0]+x[1])).view(-1,1)
    elif label=='out':
        u=(torch.tanh(x[0]+x[1])).view(-1,1) 
    else:
        raise ValueError("invalid label for u(x)")
   
    return u


def f_grad(x,label,a):
    x=x.t()
    kk=torch.tanh(x[0]+x[1])
    f=2*(-2*kk+2*kk**3).to(x.device)
    if label=='inner':  
        f=(-a*f).view(-1,1)
        
    elif label=='out':
        f=(-a*f).view(-1,1)   
    else:
        raise ValueError("invalid label for u(x)")
    
    return f

def interface_dirich(x,a):
    x=x.t()
    return ((1-a)*torch.tanh(x[0]+x[1])).view(-1,1) 

def test_data_net(args,device):  
    
    step=0.01
    x = np.arange(-1, 3+step, step)
    y = np.arange(-1, 3+step, step)
    L1=torch.tensor(args.L).to(device)
    xx,yy=np.meshgrid(x,y)
    input_x=torch.tensor(xx).view(-1,1).to(device)
    input_y=torch.tensor(yy).view(-1,1).to(device)
    input=(torch.cat((input_x,input_y),1)).float()
    index_inner=torch.where(torch.norm(input-L1,dim=1)<args.r0)[0]
    inner=input[index_inner,:]

    index_out1=torch.where(torch.norm(input-L1,dim=1)>=args.r0)[0]
    out=input[index_out1,:]
    index_out=torch.where(torch.norm(out-L1,dim=1)<=args.box)[0] # a ringing
    out=out[index_out,:]
    
    test_inner=inner.float().to(device).clone().detach()
    test_out=out.float().to(device).clone().detach()
    
    print('Totle test number of data:',test_inner.size()[0],test_out.size()[0])  
    return test_out,test_inner


def main(args):

    if torch.cuda.is_available and args.cuda:
        device='cuda'
        print('cuda is avaliable')
    else:
        device='cpu'
        
    center=torch.tensor(args.L).to(device)
    r0=args.r0
    r1=args.box
  
    ### test data,label_out,label_inner
    test_out,test_inner=test_data_net(args,device)   
    tx_in,ty_in,test_inner=data_transform(test_inner.T,device)
    tx_out,ty_out,test_out=data_transform(test_out.T,device) 
    label_out=u(test_out,'out',args.a)
    label_inner=u(test_inner,'inner',args.a)
    ### train data
    data=Data_Single(r0=r0,L=args.L,box=args.box,device=device)
    
    out,inner=data.sampleDomain_hole(args.train_domian)
    out=out.T
    inner=inner.T
    inner_b_theta,f_direction=data.sphere_bound(args.train_inner_b)
    inner_b_theta=inner_b_theta.clone().detach().requires_grad_(True)  
    x_in_b,y_in_b,input_in_b=data_transform_theta(inner_b_theta,center,r0,device) 
    x_in,y_in,input_in=data_transform(inner,device)  
    x_out,y_out,input_out=data_transform(out,device) 
    
    out_b_theta=data.sampleFromBoundary(args.train_out_b) 
    out_b_theta= out_b_theta.clone().detach().requires_grad_(True)  
    x_outb,y_outb,out_b=data_transform_theta(out_b_theta,center,r1,device)

    out_b_label=u(out_b,'out',args.a)  
    z=torch.ones(input_in_b.size()[0]).view(-1,1).to(device)

    print('out:',input_out.size())
    print('inner_b',input_in_b.size())
    print('out_b',out_b.size())
    print('input_in',input_in.size())
    
    net_inner=FNN(m=args.inner_unit).to(device) 
    net_out=FNN(m=args.out_unit).to(device)     
    optimizer=optim.Adam(itertools.chain(net_inner.parameters(),net_out.parameters()),lr=args.lr)
    result=[]
    t0=time.time()
    task={}
    task_loss={}

    scale={}
    train_loss=[]
    test_loss=[]
    scale_record=[]
    loss_history = []
    test_record=[]
    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')
    
    Mse_train_f = 1e-5
    Traing_Mse_min=1e10
    Traing_Mse_min_epoch=0
    for epoch in range(args.nepochs):      
        optimizer.zero_grad()    
        ## 1=========================================
        U1=net_inner(input_in)
        U_1x,U_1y=gradient(U1,x_in,y_in,device)  
        U_1xx=grad(U_1x,x_in,device)
        U_1yy=grad(U_1y,y_in,device)
        ff1=-(U_1xx+U_1yy)-f_grad(input_in,'inner',args.a)
        loss_in=torch.mean((ff1)**2)

        ff1_dx=grad(U_1xx+U_1yy,x_in,device)
        ff1_dy=grad(U_1xx+U_1yy,y_in,device)
        lipsch_omega1=torch.max(torch.cat((ff1_dx**2,ff1_dy**2),dim=0)) 
        del ff1_dx,ff1_dy       
        ## 2=========================================
        U1_b=net_inner(input_in_b)
        U2_b_in=net_out(input_in_b)     
        interface_dirichlet=U2_b_in-U1_b-interface_dirich(input_in_b,args.a)    
        bound_loss0=torch.mean((interface_dirichlet)**2) # interface dirichilet: 0 order
        ##
        inter_dtheta1=grad(interface_dirichlet, inner_b_theta,device)
        bound_loss1=torch.mean((inter_dtheta1)**2) #  interface dirichilet: 1 order
        d_theta1=grad(U2_b_in-U1_b, inner_b_theta,device)
        inter_dtheta2=grad(inter_dtheta1, inner_b_theta,device)
        bound_loss2=torch.mean((inter_dtheta2)**2) #  interface dirichilet: 2 order

        d_theta2=grad(d_theta1, inner_b_theta,device)
        max_d=torch.cat((d_theta1**2,d_theta2**2),dim=0)
 
        d_theta3=grad(d_theta2, inner_b_theta,device)
        max_d=torch.cat((max_d,d_theta3**2),dim=0)
  
        lipsch_interface_dirich=torch.max(max_d)
        del max_d, d_theta1, d_theta2, d_theta3,inter_dtheta1,inter_dtheta2,interface_dirichlet

        ##3=========================================
        dU1_N=torch.autograd.grad(U1_b,input_in_b, grad_outputs=z, create_graph=True)[0]     
        dU2_N=torch.autograd.grad(U2_b_in,input_in_b, grad_outputs=z, create_graph=True)[0]   
        interface_neumma=((args.a*dU2_N-dU1_N)*f_direction).sum(dim=1).view(-1,1) #  interface numman: 0 order      
        loss_out_bn0=torch.mean((interface_neumma)**2)  

        inter_ntheta1=grad(interface_neumma,inner_b_theta,device)
        loss_out_bn1=torch.mean(inter_ntheta1**2)  # interface numman: 1 order

        inter_ntheta2=grad(inter_ntheta1,inner_b_theta,device)
        max_n=torch.cat((inter_ntheta1**2,inter_ntheta2**2),dim=0)

        lipsch_interface_numman=torch.max(max_n)
        del max_n, inter_ntheta1, inter_ntheta2,interface_neumma,dU1_N,dU2_N

        #4=========================================
        U2 =net_out(input_out) 
        U_2x,U_2y=gradient(U2,x_out,y_out,device)                
        U_2xx=grad(U_2x,x_out,device)
        U_2yy=grad(U_2y,y_out,device)       
        ff2=-(U_2xx+U_2yy)*args.a-f_grad(input_out,'out',args.a)
        loss_out=torch.mean((ff2)**2)

        ff2_dx=grad((U_2xx+U_2yy),x_out,device)
        ff2_dy=grad((U_2xx+U_2yy),y_out,device)

        lipsch_omega2=torch.max(torch.cat((ff2_dx**2,ff2_dy**2),dim=0))
        del ff2_dx,ff2_dy
        ##5=========================================
        ob=net_out(out_b)
        boundary=ob-out_b_label
        loss_out_bd0=torch.mean((boundary)**2)

        outb_theta1=grad(boundary,out_b_theta,device)
        loss_out_bd1=torch.mean(outb_theta1**2)
        b_theta1=grad(ob,out_b_theta,device)     

        outb_theta2=grad(outb_theta1,out_b_theta,device)
        loss_out_bd2=torch.mean(outb_theta2**2)
     
        b_theta2=grad(b_theta1,out_b_theta,device)   
        max_b=torch.cat((b_theta1**2,b_theta2**2),dim=0)
   
        b_theta3=grad(b_theta2,out_b_theta,device)   
        max_b=torch.cat((max_b,b_theta3**2),dim=0)

        lipsch_boundary=torch.max(max_b)
        del max_b,b_theta3,b_theta1,b_theta2,outb_theta2,outb_theta1

        PINN_loss= loss_in + loss_out + (bound_loss0 + bound_loss1 +  bound_loss2)+ (loss_out_bn0 + loss_out_bn1) +(loss_out_bd0+loss_out_bd1+ loss_out_bd2)
        PINN_loss+=(lipsch_omega1+lipsch_omega2)/args.train_domian+(lipsch_boundary+lipsch_interface_numman+lipsch_interface_dirich)/args.train_domian**(1/2)/args.train_inner_b     
        PINN_loss.backward(retain_graph=True)
        optimizer.step()
                
        if (epoch+1)%args.print_num==0:
            if  (epoch+1)%args.change_epoch==0 and optimizer.param_groups[0]['lr']>1e-6:
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/2
            
            #test_out,label_out,test_inner,label_inner
        
            lipschitz_loss=lipsch_omega1+lipsch_omega2+lipsch_boundary+lipsch_interface_numman+lipsch_interface_dirich 
            print('Epoch                         : ',epoch+1)   
            print('Training MSE, Lipschitz_loss  :',PINN_loss.item(),lipschitz_loss)                        
                
            L_in=net_inner(test_inner)-label_inner
            L_out=net_out(test_out)-label_out
            # L_2 error calculate
            L2_in=torch.sqrt(torch.nn.MSELoss()(L_in,L_in*0))
            L2_out=torch.sqrt(torch.nn.MSELoss()(L_out,L_out*0))
            # H_2 error calculate
            dx_i=grad(L_in,tx_in,device)
            dy_i=grad(L_in,ty_in,device)
            dxx_i=grad(dx_i,tx_in,device)
            dyy_i=grad(dy_i,ty_in,device)
            dxy_i=grad(dy_i,tx_in,device)
            H2_in=torch.sqrt(torch.mean(L_in**2+dx_i**2+dy_i**2+dxx_i**2+dyy_i**2+2*dxy_i**2))

            dx_o=grad(L_out,tx_out,device)
            dy_o=grad(L_out,ty_out,device)
            dxx_o=grad(dx_o,tx_out,device)
            dyy_o=grad(dy_o,ty_out,device)
            dxy_o=grad(dy_o,tx_out,device)
            H2_out=torch.sqrt(torch.mean(L_out**2+dx_o**2+dy_o**2+dxx_o**2+dyy_o**2+2*dxy_o**2))

            print('Test numbers                  :',test_inner.size(),test_out.size())
            print('Test L2                       :',L2_in.item(),',',L2_out.item())
            print('Test H2                       :',H2_in.item(),',',H2_out.item())                      
            print('*****************************************************')  

    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')
    torch.save(net_inner, 'outputs/'+args.filename+'/model/inner.pkl')
    torch.save(net_out, 'outputs/'+args.filename+'/model/out.pkl')
    print('training_down!')
        
if __name__ == '__main__':
    torch.cuda.set_device(0)
    number=100
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',type=str, default='results')
    parser.add_argument('--train_inner_b', type=int, default=10*int(np.sqrt(number)))
    parser.add_argument('--train_domian', type=int, default=number)
    parser.add_argument('--train_out_b', type=int, default=10*int(np.sqrt(number)))
    parser.add_argument('--inner_unit', type=int, default=200)
    parser.add_argument('--out_unit', type=int, default=200)
    parser.add_argument('--print_num', type=int, default=100)
    parser.add_argument('--nepochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--r0', type=float, default=1)
    parser.add_argument('--a', type=float, default=2)
    parser.add_argument('--L', type=list, default=[1,1])
    parser.add_argument('--box', type=list, default=2)
    parser.add_argument('--change_epoch', type=int, default=1000)
    parser.add_argument('--save', type=str, default=False)
    args = parser.parse_args()
    main(args)


           
