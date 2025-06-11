import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch
import math
import random
from pathlib import Path

#from rmsnorm_torch import RMSNorm

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class Unimodal_Multivariate_Gaussian:
    def __init__(self,dim):
        mean = torch.zeros(dim)
        self.dist = torch.distributions.Normal(mean,torch.ones_like(mean))
        self.dim = dim
    
    def num_data_dim(self):
        return self.dim
    
    def sample(self,n):
        if isinstance(n,int):
            return self.dist.sample((n,))
        elif isinstance(n,tuple):
            return self.dist.sample(n)
        else:
            raise NotImplementedError

class Bimodal_Multivariate_Gaussian:
    def __init__(self,dim):
        mean = torch.ones(2,dim)
        mean[1,:] = -mean[1,:]
        std = 0.25*torch.ones_like(mean)
        comp = torch.distributions.Independent(torch.distributions.Normal(mean,std),1)
        mix = torch.distributions.Categorical(torch.tensor([0.5,0.5]))
        self.dist = torch.distributions.MixtureSameFamily(mix,comp)
        self.dim = dim
    
    def num_data_dim(self):
        return self.dim

    def sample(self,n):
        if isinstance(n,int):
            return self.dist.sample((n,))
        elif isinstance(n,tuple):
            return self.dist.sample(n)
        else:
            raise NotImplementedError

class Multimodal_Gaussian_2D:
    def __init__(self):
        D = 10.0
        probs = torch.tensor([1/6 for _ in range(6)])
        means = torch.tensor([
                [D * np.sqrt(3) / 2., D / 2.], 
                [-D * np.sqrt(3) / 2., D / 2.], 
                [0.0, -D],
                [D * np.sqrt(3) / 2., - D / 2.],
                [-D * np.sqrt(3) / 2., - D / 2.],
                [0.0, D]]).float()
        covs = 0.5*torch.stack([torch.eye(2) for _ in range(6)])
        comp = torch.distributions.Independent(torch.distributions.MultivariateNormal(means, covs),0)
        mix = torch.distributions.Categorical(probs)
        self.dist = torch.distributions.MixtureSameFamily(mix,comp)

    def num_data_dim(self):
        return 2

    def sample(self,n):
        if isinstance(n,int):
            return self.dist.sample((n,))
        elif isinstance(n,tuple):
            return self.dist.sample(n)
        else:
            raise NotImplementedError

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        assert(self.dim%2==0)

    def forward(self,x):
        device = x.device
        half_dim = self.dim//2
        emb = math.log(10000)/(half_dim-1)
        emb = torch.exp(torch.arange(half_dim,device=device)*(-emb)).view((1,)*x.dim()+(-1,))
        emb = x[...,None]*emb
        emb = torch.cat((emb.sin(),emb.cos()),dim=-1)
        return emb

class Net(torch.nn.Module):
    def __init__(self,num_data_dim):
        super().__init__()
        dim = 128
        fourier_dim = dim
        data_dim = dim
        out_dim = num_data_dim

        self.mlp = torch.nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            torch.nn.Linear(data_dim,data_dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(num_data_dim*data_dim,num_data_dim*data_dim),
            torch.nn.GELU(),
            torch.nn.Linear(num_data_dim*data_dim,out_dim),
        )

    def forward(self, xi):
        return self.mlp(xi)
    

def main():
    max_steps = 500
    num_data_dim = 1
    n = 200
    m = 100
    alpha = 1.0
    beta = 0.1

    source_distribution = Unimodal_Multivariate_Gaussian(num_data_dim)
    # target_distribution = Multimodal_Gaussian_2D()
    target_distribution = Bimodal_Multivariate_Gaussian(num_data_dim)
    num_data_dim = target_distribution.num_data_dim()

    net = Net(num_data_dim)
    optimizer = torch.optim.AdamW(net.parameters(),lr=1e-3)
    loss_vals = np.zeros((max_steps,))

    with tqdm(total=max_steps) as pbar:
        for step in range(max_steps):

            x_1 = target_distribution.sample(n)
            xi = source_distribution.sample((n,m))

            samples = net(xi.view(-1,num_data_dim)).view(xi.shape)

            l_data = torch.sqrt(1e-6+torch.sum((x_1[:,None,...] - samples)**2,dim=-1))**beta
            loss = torch.sqrt(1e-6+torch.sum((samples[:,None,...] - samples[...,None,:])**2,dim=-1))**beta
            loss = torch.mean(l_data - alpha*torch.sum(loss,dim=-1)/2/(m-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_vals[step] = loss.item()
            pbar.set_description(f"loss: {loss_vals[step]:.4f}")
            pbar.update(1)
    
    plt.plot(loss_vals)
    plt.savefig("main.png")

    with torch.inference_mode():
        x_0 = torch.randn((10000,num_data_dim))
        samples = net(x_0)
        plt.clf()
        plt.hist(samples,bins=100, label="samples")
        plt.hist(target_distribution.sample(samples.shape[0]),bins=100, label="target", alpha=0.5)
        plt.legend()
        plt.savefig("samples.png")

if __name__=="__main__":
    main()