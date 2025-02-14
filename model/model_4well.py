import os
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from .ae import AE
from data import Datasets
from .codebook import CodeBook
from .dynamics import EnergyDynamics4Well, EnergyPredictor
from utils import test_4_well



def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, 0)



class PESLA_4_Well(nn.Module):
    def __init__(self, conf):
        super().__init__()
        
        # Model args
        code_dim, K = conf.PESLA_4_Well.code_dim, conf.PESLA_4_Well.K
        self.conf = conf
        
        # VQ-VAE
        self.encoder, self.decoder = AE(conf)
        self.codebook = CodeBook(K, code_dim, conf.PESLA_4_Well.softmax_temperature, init='uniform')
        
        # Energy Predictor
        self.energyNet = EnergyPredictor(conf.PESLA_4_Well.code_dim)
        
        # Device
        self.to(conf.device)
        
        # Initialize weights
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        code_idx = self.codebook(z_e_x) # (B,)
        return code_idx
    
    def Energy(self, x):
        z_e_x = self.encoder(x)
        return self.energyNet(z_e_x)

    def forward(self, x, test=False):
        # x: (B, C)
        z_e_x = self.encoder(x) # (B, D)
        greedy = self.conf.PESLA_4_Well.greedy_train or test
        z_q_x_st, z_q_x, x_code_idx = self.codebook.straight_through(z_e_x, greedy=greedy) # (B, D)
        x_recons, mu, std = self.decoder(z_q_x_st) # (B, C)

        return x_recons, z_e_x, z_q_x, x_code_idx, mu, std
    
    def predict(self, x):
        # x: (B, C)
        z_e_x = self.encoder(x) # (B, D)
        _, _, x_code_idx = self.codebook.straight_through(z_e_x, greedy=True) # (B, D)
        h1, p1_logit = self.dynamics.evolve_one_step(x_code_idx, lag=self.conf._4_Well.lag)
        return h1, p1_logit
    
    def H(self, x):
        # x: (B, C)
        z_e_x = self.encoder(x)
        _, _, x_code_idx = self.codebook.straight_through(z_e_x, greedy=True) # (B, D)
        h = self.dynamics.H(x_code_idx) # (B, D)
        return h
    
    def sample(self, init_state, n_samples):        
        # init_state: (B, C)
        # Encode
        z_e_x = self.encoder(init_state) # (B, D)
        code_idx = self.codebook(z_e_x) # (B,)

        # Transition
        P0 = torch.zeros((len(init_state), self.codebook.K), device=self.conf.device) # (B, N)
        P0[torch.arange(len(init_state)), code_idx] = 1.0
        sample_traj, sample_traj_code_idx, sample_traj_P = [init_state.unsqueeze(0)], [code_idx.unsqueeze(0)], [P0.unsqueeze(0)]
        self.eval()
        for _ in tqdm(range(n_samples)):
            _, P_logit_hat = self.predict(sample_traj[-1][0])
            P = F.softmax(P_logit_hat, dim=-1) # (B, N)
            
            code_idx = torch.multinomial(P, 1).view(-1) # (B,)
            code = self.codebook.lookup(code_idx) # (B, D)
            state, _, _ = self.decoder(code) # (1, B, C)
            
            sample_traj.append(state.unsqueeze(0))
            sample_traj_code_idx.append(code_idx.unsqueeze(0))
            sample_traj_P.append(P.unsqueeze(0))
        
        sample_traj = torch.cat(sample_traj, dim=0).permute(1, 0, 2).contiguous() # (B, n_samples, C)
        sample_traj_code_idx = torch.cat(sample_traj_code_idx, dim=0).permute(1, 0).contiguous() # (B, n_samples)
        sample_traj_P = torch.cat(sample_traj_P, dim=0).permute(1, 0, 2).contiguous() # (B, n_samples, N)
        return sample_traj, sample_traj_code_idx, sample_traj_P
    
    def gaussian_nll(self, x, mu, std):
        return 0.5*((x-mu).pow(2)/(std.pow(2)+1e-8) + 2*torch.log(std+1e-8) + np.log(2*np.pi)).sum(-1).mean()
    
    def boltzmann_kldiv(self, code_idx, k=1.0, T=1.0):
        # code_idx: (B,)

        unique, counts = code_idx.unique(return_counts=True)
        q = counts.float() / code_idx.size(0)
        
        if len(unique) < 5:
            return torch.tensor(0.0, device=self.conf.device)

        energy = self.energyNet(self.codebook.lookup(unique).detach())
        p = torch.exp(-energy/(k * T)) / torch.exp(-energy/(k * T)).sum()
        
        return (p * torch.log(p / q.view(-1,1))).sum()
    
    def fit(self):
        batch_size = self.conf._4well_phase1.batch_size
        # Dataset
        train_dataset = Datasets[self.conf.system](self.conf, mode='phase1_train')
        val_dataset = Datasets[self.conf.system](self.conf, mode='phase1_val')
        train_loader, val_loader = train_dataset.getLoader(batch_size, shuffle=True), val_dataset.getLoader(batch_size, shuffle=False)
        self.fit_phase1(train_loader, val_loader)
        
        batch_size = self.conf._4well_phase2.batch_size
        train_dataset = Datasets[self.conf.system](self.conf, mode='phase2_train')
        val_dataset = Datasets[self.conf.system](self.conf, mode='phase2_val')
        train_loader, val_loader = train_dataset.getLoader(batch_size, shuffle=True), val_dataset.getLoader(batch_size, shuffle=False)
        self.fit_phase2(train_loader, val_loader)
    
    def fit_phase1(self, train_loader, val_loader):
        if os.path.exists(self.conf.log_dir+'phase1.pt'):
            self.load_state_dict(torch.load(self.conf.log_dir+'phase1.pt'))
            print('Load pre-trained model (Phase1) ...')
            return

        os.makedirs(self.conf.log_dir+'val/phase1/', exist_ok=True)
        print(f'Training {self.conf.model} model (Phase1) ...')
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.codebook.parameters()) + list(self.energyNet.parameters()), 
            lr=self.conf._4well_phase1.lr
        )
        scheduler = ExponentialLR(optimizer, gamma=self.conf._4well_phase1.lr_decay)
        
        train_loss_list, val_loss_list, hit_num_list = [], [], []
        for epoch in range(1, self.conf._4well_phase1.max_epoch+1):
            
            train_loss = 0.0
            self.train()
            
            for i, (X, Y) in enumerate(train_loader):
                x_recons, z_e_x, z_q_x, X_code_idx, mu, std = self(X)
                
                # Reconstruction loss
                loss_recons = self.gaussian_nll(X, mu, std)
                # Vector quantization objective
                loss_vq = ((z_q_x-z_e_x.detach()).pow(2).sum(-1)).mean()
                # Commitment objective
                loss_commit = ((z_q_x.detach()-z_e_x).pow(2).sum(-1)).mean()
                # Pretrain energy
                loss_l = self.boltzmann_kldiv(X_code_idx)
                
                loss = loss_recons + loss_vq + loss_commit + loss_l
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss_recons.item()
            
            print(f'\rEpoch[{epoch}/{self.conf._4well_phase1.max_epoch}] nll={loss_recons.item():.4f} vq={loss_vq.item():.4f} commit={loss_commit.item():.4f}', end='')
            train_loss_list.append([epoch, train_loss / len(train_loader)])
            
            if epoch % self.conf._4well_phase1.lr_step == 0:
                scheduler.step()
            
            if epoch % self.conf._4well_phase1.val_interval == 0:
                self.eval()
                val_loss, hit_codes, xy_true, xy_pred = 0., [], [], []
                for i, (X, Y) in enumerate(val_loader):
                    x_recons, z_e_x, z_q_x, code_idx, mu, std = self(X, test=True)
                    
                    loss_recons = self.gaussian_nll(X, mu, std)
                    val_loss += loss_recons.item()
                    
                    xy_true.extend(X.cpu().detach().numpy().tolist())
                    xy_pred.extend(x_recons.cpu().detach().numpy().tolist())
                    code_idx = self.encode(Y)
                    hit_codes.extend(code_idx.view(-1).cpu().detach().numpy().tolist())
                
                # hit num
                hit_num = np.unique(hit_codes).size
                hit_num_list.append(hit_num)
                
                print(f'\nEpoch[{epoch}/{self.conf._4well_phase1.max_epoch}] val loss={val_loss/len(val_loader):.4f}')
                val_loss_list.append([epoch, val_loss/len(val_loader)])
                
                v_min, v_max = X.min().item(), X.max().item()
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(np.array(xy_true)[:,0], np.array(xy_pred)[:,0], s=1)
                plt.plot([v_min, v_max], [v_min, v_max], 'r--')
                plt.xlabel('x_true')
                plt.ylabel('x_pred')
                plt.subplot(1, 2, 2)
                plt.scatter(np.array(xy_true)[:,1], np.array(xy_pred)[:,1], s=1)
                plt.plot([v_min, v_max], [v_min, v_max], 'r--')
                plt.xlabel('y_true')
                plt.ylabel('y_pred')
                plt.tight_layout()
                plt.savefig(self.conf.log_dir+f'val/phase1/regression_{epoch}.png', dpi=300)
        
        # Save Phase1 model
        torch.save(self.state_dict(), self.conf.log_dir+'phase1.pt')
    
    
    def fit_phase2(self, train_loader, val_loader):
        if os.path.exists(self.conf.log_dir+f'lag{self.conf._4_Well.lag}-phase2.pickle'):
            with open(self.conf.log_dir+f'lag{self.conf._4_Well.lag}-phase2.pickle', 'rb') as f:
                [self.dynamics, self.codebook, self.energyNet] = pickle.load(f)
                self.dynamics.pe = self.dynamics.pe.to(self.conf.device)
                self.dynamics.node_energy = self.dynamics.node_energy.to(self.conf.device)
                self.dynamics.A = self.dynamics.A.to(self.conf.device)
                self.dynamics.odenet.E = self.dynamics.odenet.E.to(self.conf.device)
                self.to(self.conf.device)
            print('Load pre-trained model (Phase2) ...')
            return
        else:
            print(f'Training {self.conf.model} model (Phase2) ...')
        
        hit_code_idx = []
        for loader in [train_loader, val_loader]:
            for i, (X, Y) in enumerate(loader):
                _, _, _, X_code_idx, _, _ = self(X)
                hit_code_idx.extend(X_code_idx.view(-1).cpu().detach().numpy().tolist())
        hit_code_idx = np.unique(hit_code_idx)
        
        self.codebook.embedding.weight.data = self.codebook.embedding.weight.data[hit_code_idx]
        self.codebook.K = len(hit_code_idx)
        
        node_features = self.codebook.embedding.weight.data.detach().cpu().numpy()
        node_energy = self.energyNet(self.codebook.embedding.weight.data).detach()
        self.dynamics = EnergyDynamics4Well(self.conf, node_features, node_energy).to(self.conf.device)
        
        optimizer = torch.optim.Adam(
            list(self.dynamics.parameters()) + list(self.energyNet.parameters()), 
            lr=self.conf._4well_phase2.lr
        )
        scheduler = ExponentialLR(optimizer, gamma=self.conf._4well_phase2.lr_decay)
        
        train_loss_list, val_loss_list = [], []
        for epoch in range(1, self.conf._4well_phase2.max_epoch+1):
            
            train_loss = 0.0
            self.train()
            
            for i, (X, Y) in enumerate(train_loader):
                H_hat, P_logit_hat = self.predict(X)
                H = self.H(Y)
                y_code_idx = self.encode(Y)
                
                loss_h = F.mse_loss(H_hat, H.detach())
                loss_p = F.cross_entropy(P_logit_hat, y_code_idx)
                loss_l = self.boltzmann_kldiv(y_code_idx)
                loss = loss_h + loss_p + loss_l
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            print(f'\rEpoch[{epoch}/{self.conf._4well_phase2.max_epoch}] loss_h={loss_h.item():.4f} loss_p={loss_p.item():.4f} loss_l={loss_l.item():.4f}', end='')
            
            train_loss_list.append([epoch, train_loss / len(train_loader)])
            
            if epoch % self.conf._4well_phase2.lr_step == 0:
                scheduler.step()
                
            if epoch % self.conf._4well_phase2.val_interval == 0:
                self.eval()
                val_loss = 0.
                for i, (X, Y) in enumerate(val_loader):
                    H_hat, P_logit_hat = self.predict(X)
                    H = self.H(Y)
                    y_code_idx = self.encode(Y)
                
                    loss_h = F.mse_loss(H_hat, H)
                    loss_p = F.cross_entropy(P_logit_hat, y_code_idx)
                    loss = loss_h + loss_p
                    
                    val_loss += loss.item()
                print(f'\nEpoch[{epoch}/{self.conf._4well_phase2.max_epoch}] val loss_h={loss_h.item():.4f} val loss_p={loss_p.item():.4f}')
                val_loss_list.append([epoch, val_loss/len(val_loader)])
            
        # Save Phase2 model
        with open(self.conf.log_dir+f'lag{self.conf._4_Well.lag}-phase2.pickle', 'wb') as f:
            pickle.dump([self.dynamics, self.codebook, self.energyNet], f)
    
    
    def test(self):
        
        os.makedirs(self.conf.log_dir+'test/', exist_ok=True)
        self.eval()
        self.to(self.conf.device)
        
        # Sample Generation
        traj_num = 10
        n_samples = 100000
        grid_num = 5
        test_4_well(self, traj_num, n_samples, grid_num, lag=self.conf._4_Well.lag)