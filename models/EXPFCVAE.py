import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from torch.utils import data
from collections import defaultdict

class EXPFCVAE(nn.Module):
    def __init__(self, training_set, args, abs_, optimize_both_exp=True, capacity1=50, capacity2=25, dropout1=.75, dropout2=.5, batchnorm=True):
        super(EXPFCVAE, self).__init__()
        #Dropout in encoder (UNTIL Z IS USED)
        #Softplus or batchnorm on log_var
        #Stack up all spikes in one big vector, make everything local, define 1-2 layer decoder and transforms it, then
        #exponential
        self.capacity1 = capacity1
        self.capacity2 = capacity2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.optimize_both_exp = optimize_both_exp
        self.num_spikes = len(training_set)
        num_amps = training_set[0][0].shape[0]
        
        self.fc1 = nn.Linear(num_amps*2, self.capacity1)
        self.fc2 = nn.Linear(self.capacity1, self.capacity2)
        self.fc_mean = nn.Linear(self.capacity2, 3)
        self.fc_cov = nn.Linear(self.capacity2, 9)
        
        self.batchnorm = batchnorm
        if(batchnorm):
            self.batchnorm1 = nn.BatchNorm1d(self.capacity1)
            self.batchnorm2 = nn.BatchNorm1d(self.capacity2)
        
        a_s = abs_[:,0]
        b_s = abs_[:,1]
        abs_ = torch.from_numpy(abs_).float()
        a_s = torch.from_numpy(a_s).float()
        self.b_s = torch.from_numpy(b_s).float()
        
        if(args.cuda):
            if(optimize_both_exp):
                self.exps = torch.tensor(abs_, requires_grad=True, device="cuda")
            else:   
                self.exps = torch.tensor(a_s, requires_grad=True, device="cuda")
        else:
            if(optimize_both_exp):
                self.exps_0 = torch.tensor(abs_)
                self.exps = torch.tensor(abs_, requires_grad=True)
            else:   
                self.exps_0 = torch.tensor(a_s)
                self.exps = torch.tensor(a_s, requires_grad=True)
                
        
    def encode(self, amps):
        #Use soft-plus instead of log_var (gradient of soft-plus nicer)
        if(self.batchnorm):
            x = self.batchnorm1(self.fc1(torch.flatten(amps, start_dim=1)))
        else:
            x = self.fc1(torch.flatten(amps, start_dim=1))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout1) 
        if(self.batchnorm):
            x = self.batchnorm2(self.fc2(x))
        else:
            x = self.fc2(x)   
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout2)
        xyz_mu = self.fc_mean(x)
        cov_xyz = self.fc_cov(x)
        
        A = cov_xyz.view(cov_xyz.shape[0], 3, 3)
        
        return xyz_mu, A

    def reparameterize_mvn(self, mu, A):
        mu = mu.view(mu.shape[0], mu.shape[1], 1)
        eps = torch.randn_like(mu)
        return A.bmm(eps).add_(mu)
    
    def getTensorDistances(self, n_loc, ch_locs):
        n_loc = n_loc.view(n_loc.shape[0], 1, n_loc.shape[1])
        subtract = (n_loc - ch_locs)**2
        summed = torch.sum(subtract, dim=2) 
        return torch.sqrt(summed)
    
    def decode(self, sampled_n_loc, ch_locs, spike_ids):
        distances = self.getTensorDistances(sampled_n_loc, ch_locs)#
        #Exponential observation model with model parameters a, b
        if(self.optimize_both_exp):
            a_exps = torch.index_select(self.exps, 0, spike_ids)[:,0]
            b_exps = torch.index_select(self.exps, 0, spike_ids)[:,1]
        else:   
            a_exps = torch.index_select(self.exps, 0, spike_ids)
            b_exps = torch.index_select(self.b_s, 0, spike_ids)
        a_exps = a_exps.view(a_exps.shape[0], 1)
        b_exps = b_exps.view(b_exps.shape[0], 1)
        recon_amps = -torch.exp(distances*b_exps + a_exps)
        return recon_amps

    def forward(self, amps, ch_locs, spike_ids):
        xyz_mu, A = self.encode(amps)

        if(self.training):
            xyz_sample = self.reparameterize_mvn(xyz_mu, A)
        else:
            xyz_sample = xyz_mu
            
        recon_amps = self.decode(xyz_sample, ch_locs, spike_ids)
        return recon_amps, xyz_mu, A
    
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_amps, t_amps, xyz_mu, A, MSEs, epoch, device):
    EPSILON = 1e-5
    
    batch_size = recon_amps.shape[0]
    
    
    mu_q = xyz_mu
    cov_q = torch.bmm(A, torch.transpose(A, 1, 2)) + torch.Tensor(torch.eye(3))*EPSILON
#     print(cov_q)
    
    mu_p = torch.zeros((batch_size, 3))
    cov_p = 80*torch.Tensor(torch.eye(3)).view(9,1).repeat(batch_size,1).view(batch_size, 3,3)
    
    mvn_q = torch.distributions.multivariate_normal.MultivariateNormal(mu_q, cov_q)
    mvn_p = torch.distributions.multivariate_normal.MultivariateNormal(mu_p, cov_p)
    
    KLD = torch.sum(torch.distributions.kl_divergence(mvn_q, mvn_p))/batch_size
    MSE = torch.sum(F.mse_loss(recon_amps, t_amps[:,:,0], reduction='none')*t_amps[:,:,1])/batch_size

    return MSE + KLD

def train(model, device, args, optimizer, train_loader, epoch, train_losses, distance_diffs, a_list, b_list, MSEs, xklds, yklds, zklds):
    model.train()
    train_loss = 0
    for batch_idx, (t_amps, t_ch_locs, center_loc, spike_ids, _, _) in enumerate(train_loader):
        t_amps = t_amps.to(device) #torch.Size([200, 49, 2])
        t_ch_locs = t_ch_locs.to(device) #torch.Size([200, 49, 3])
        spike_ids = spike_ids.to(device)
        optimizer.zero_grad()
        recon_amps, xyz_mu, A = model(t_amps, t_ch_locs, spike_ids)
        loss = loss_function(recon_amps, t_amps, xyz_mu, A, MSEs, epoch, device)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(t_amps[0].shape), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(t_amps[0].shape)))
            train_losses.append(train_loss)
#             distance_diffs.append(dist_diff)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
    return model