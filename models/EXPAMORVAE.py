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

class EXPAMORVAE(nn.Module):
    def __init__(self, training_set, args, abs_, optimize_both_exp=True, capacity1=50, capacity2=25, dropout1=.75, dropout2=.5, batchnorm=True):
        super(EXPAMORVAE, self).__init__()
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
        self.fc_var = nn.Linear(self.capacity2, 3)
        self.fc2_mean = nn.Linear(self.capacity2, 1)
        self.fc2_var = nn.Linear(self.capacity2, 1)
        
        self.batchnorm = batchnorm
        if(batchnorm):
            self.batchnorm1 = nn.BatchNorm1d(self.capacity1)
            self.batchnorm2 = nn.BatchNorm1d(self.capacity2)
        
        a_s = abs_[:,0]
        b_s = abs_[:,1]
        self.a_s = torch.from_numpy(a_s).float()
        self.b_s = torch.from_numpy(b_s).float()
        
    def encode(self, amps):
        #Use soft-plus instead of log_var (gradient of soft-plus nicer)
        EPSILON = 1e-6
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
        xyz_var = F.softplus(self.fc_var(x)) + EPSILON
        
        a_mu = F.softplus(self.fc2_mean(x)) + EPSILON
        a_var = F.softplus(self.fc2_var(x)) + EPSILON
        
        x_mu = xyz_mu[:,0]
        x_mu = x_mu.view(x_mu.shape[0], 1)
        x_var = xyz_var[:,0]
        x_var = x_var.view(x_var.shape[0], 1)
        
        y_mu = xyz_mu[:,1]
        y_mu = y_mu.view(y_mu.shape[0], 1)
        y_var = xyz_var[:,1]
        y_var = y_var.view(y_var.shape[0], 1)
        
        z_mu = xyz_mu[:,2]
        z_mu = z_mu.view(z_mu.shape[0], 1)
        z_var = xyz_var[:,2]
        z_var = z_var.view(z_var.shape[0], 1)
        z_var = z_var
        
        return x_mu, x_var, y_mu, y_var, z_mu, z_var, a_mu, a_var

    def reparameterize_normal(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def getTensorDistances(self, n_loc, ch_locs):
        n_loc = n_loc.view(n_loc.shape[0], 1, n_loc.shape[1])
        subtract = (n_loc - ch_locs)**2
        summed = torch.sum(subtract, dim=2) 
        return torch.sqrt(summed)
    
    def decode(self, sampled_n_loc, sampled_a_exps, ch_locs, spike_ids):
        distances = self.getTensorDistances(sampled_n_loc, ch_locs)#
        #Exponential observation model with model parameters a, b
        b_exps = torch.index_select(self.b_s, 0, spike_ids)
        b_exps = b_exps.view(b_exps.shape[0], 1)
        recon_amps = -torch.exp(distances*b_exps + sampled_a_exps)
        return recon_amps

    def forward(self, amps, ch_locs, spike_ids):
        x_mu, x_var, y_mu, y_var, z_mu, z_var, a_mu, a_var = self.encode(amps)
        if(self.training):
            x_sample = self.reparameterize_normal(x_mu, x_var)
            y_sample = self.reparameterize_normal(y_mu, y_var)
            z_sample = self.reparameterize_normal(z_mu, z_var)
            sampled_a_exps = self.reparameterize_normal(a_mu, a_var)
        else:
            x_sample = x_mu
            y_sample = y_mu
            z_sample = z_mu
            sampled_a_exps = a_mu
        sampled_n_loc = torch.cat((x_sample, y_sample, z_sample), 1)
        recon_amps = self.decode(sampled_n_loc, sampled_a_exps, ch_locs, spike_ids)
        return recon_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var, a_mu, a_var
    
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_amps, t_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var, a_mu, a_var, a_mu0, epoch, device, ):
    #Leave variance at 1 and send plots of KL divergence to Akash
    #Change to normal gaussian loss with arbitrary variance
    batch_size = recon_amps.shape[0]
    MSE = torch.sum(F.mse_loss(recon_amps, t_amps[:,:,0], reduction='none')*t_amps[:,:,1])/batch_size

    m_qx = x_mu
    m_px = torch.zeros(m_qx.shape)
    var_qx = x_var
    var_px = torch.zeros(var_qx.shape) + 80**2
    KLD_x = kl_divergence_normal(m_qx, var_qx, m_px, var_px)/batch_size

    m_qy = y_mu
    m_py = torch.zeros(m_qx.shape)
    var_qy = y_var
    var_py = torch.zeros(var_qx.shape) + 80**2
    KLD_y = kl_divergence_normal(m_qy, var_qy, m_py, var_py)/batch_size


    m_qz = z_mu
    m_pz = torch.zeros(m_qx.shape)
    var_qz = z_var
    var_pz = torch.zeros(var_qx.shape) + 80**2
    KLD_z = kl_divergence_normal(m_qz, var_qz, m_pz, var_pz)/batch_size
    
    m_qa = a_mu
    m_pa = a_mu0.float()
    var_qa = a_var
    var_pa = torch.zeros(var_qa.shape) + 10**2
    KLD_a = kl_divergence_normal(m_qa, var_qa, m_pa, var_pa)/batch_size
    
    return MSE + KLD_x + KLD_y + KLD_z + KLD_a

def kl_divergence_normal(mu_q, var_q, mu_p, var_p):
    kld = torch.sum(0.5*(torch.log(var_p) - torch.log(var_q)) + torch.div(var_q + (mu_q - mu_p)**2, 
                                                            2*var_p) - 0.5)
    return kld

def train(model, device, args, optimizer, train_loader, epoch, train_losses):
    model.train()
    train_loss = 0
    for batch_idx, (t_amps, t_ch_locs, center_loc, spike_ids, _, _) in enumerate(train_loader):
        t_amps = t_amps.to(device) #torch.Size([200, 49, 2])
        t_ch_locs = t_ch_locs.to(device) #torch.Size([200, 49, 3])
        spike_ids = spike_ids.to(device)
        optimizer.zero_grad()
        recon_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var, a_mu, a_var = model(t_amps, t_ch_locs, spike_ids)
        a_mu0 = torch.index_select(model.a_s, 0, spike_ids)
        a_mu0 = a_mu0.view(a_mu0.shape[0], 1)
        loss = loss_function(recon_amps, t_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var, a_mu, a_var, spike_ids, epoch, device)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(t_amps[0].shape), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(t_amps[0].shape)))
            train_losses.append(train_loss)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
    return model