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
from .dynamics import EnergyDynamicsSSWM, EnergyPredictor
from utils import test_sswm



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



class PESLA_SSWM(nn.Module):
    def __init__(self, conf):
        super().__init__()
        
        # Model args
        code_dim, K = conf.PESLA_SSWM.code_dim, conf.PESLA_SSWM.K
        self.conf = conf
        
        # VQ-VAE
        self.encoder, self.decoder = AE(conf)
        self.codebook = CodeBook(K, code_dim, conf.PESLA_SSWM.softmax_temperature, init='uniform')
        
        # Energy Predictor
        self.energyNet = EnergyPredictor(conf.PESLA_SSWM.code_dim)
        
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
        # x: (B, loci, N_states)
        z_e_x = self.encoder(x) # (B, D)
        greedy = self.conf.PESLA_SSWM.greedy_train or test
        z_q_x_st, z_q_x, x_code_idx = self.codebook.straight_through(z_e_x, greedy=greedy) # (B, D)
        x_logits = self.decoder(z_q_x_st) # (B, loci, N_states)

        return x_logits, z_e_x, z_q_x, x_code_idx
    
    def predict(self, x):
        # x: (B, loci, N_states)
        z_e_x = self.encoder(x) # (B, D)
        _, _, x_code_idx = self.codebook.straight_through(z_e_x, greedy=True) # (B, D)
        h1, p1_logit = self.dynamics.evolve_one_step(x_code_idx, lag=self.conf.SSWM.lag)
        return h1, p1_logit
    
    def H(self, x):
        # x: (B, loci, N_states)
        z_e_x = self.encoder(x) # (B, D)
        _, _, x_code_idx = self.codebook.straight_through(z_e_x, greedy=True) # (B, D)
        h = self.dynamics.H(x_code_idx) # (B, D)
        return h
    
    def sample(self, init_state, n_samples):        
        # init_state: (B, loci, N_states)
        # Encode
        z_e_x = self.encoder(init_state) # (B, D)
        code_idx = self.codebook(z_e_x) # (B,)
        
        # Transition
        P0 = torch.zeros((len(init_state), self.codebook.K), device=self.conf.device) # (B, N)
        P0[torch.arange(len(init_state)), code_idx] = 1.0
        sample_traj, sample_traj_code_idx, sample_traj_P = [], [], []
        for _ in tqdm(range(n_samples)):
            _, P_logit_hat = self.predict(init_state)
            P = F.softmax(P_logit_hat, dim=-1) # (B, N)
            
            code_idx = torch.multinomial(P, 1).view(-1) # (B,)
            code = self.codebook.lookup(code_idx) # (B, D)
            logits = self.decoder(code) # (B, loci, N_states)
            init_state = logits
            
            # Sample
            state = torch.zeros((init_state.shape[0], init_state.shape[1]), device=self.conf.device)
            for i in range(init_state.size(1)):
                state[:, i] = F.softmax(logits[:, i], dim=-1).multinomial(1).squeeze(-1)
            
            sample_traj.append(state.unsqueeze(0).cpu().detach().numpy()) # (1, B, loci)
        
        sample_traj = np.concatenate(sample_traj, axis=0).transpose(1, 0, 2) # (n_samples, B, loci)
        return sample_traj
    
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
        batch_size = self.conf.sswm_phase1.batch_size
        train_dataset = Datasets[self.conf.system](self.conf, mode='phase1_train')
        val_dataset = Datasets[self.conf.system](self.conf, mode='phase1_val')
        train_loader, val_loader = train_dataset.getLoader(batch_size, shuffle=True), val_dataset.getLoader(batch_size, shuffle=False)
        self.fit_phase1(train_loader, val_loader)
        
        batch_size = self.conf.sswm_phase2.batch_size
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
            lr=self.conf.sswm_phase1.lr
        )        
        scheduler = ExponentialLR(optimizer, gamma=self.conf.sswm_phase1.lr_decay)
        
        train_loss_list, val_loss_list, hit_num_list = [], [], []
        for epoch in range(1, self.conf.sswm_phase1.max_epoch+1):
            
            train_loss = 0.0
            self.train()

            for i, (X, Y) in enumerate(train_loader):
                # X: (B, loci, N_states)
                x_logits, z_e_x, z_q_x, x_code_idx = self(X) # (B, loci, N_states), (B, D), (B, D), (B,)
                
                # Reconstruction loss
                loss_recons = F.cross_entropy(torch.concat([x_logits[:,0], x_logits[:,1]], dim=1), torch.concat([X[:,0], X[:,1]], dim=1))
                # Vector quantization objective
                loss_vq = ((z_q_x-z_e_x.detach()).pow(2).sum(-1)).mean()
                # Commitment objective
                loss_commit = ((z_q_x.detach()-z_e_x).pow(2).sum(-1)).mean()
                # Energy
                loss_energy = self.boltzmann_kldiv(x_code_idx)
                
                loss = loss_recons + loss_vq + loss_commit + loss_energy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss_recons.item()
            
            print(f'\rEpoch[{epoch}/{self.conf.sswm_phase1.max_epoch}] recons={loss_recons.item():.4f} vq={loss_vq.item():.4f} energy={loss_energy.item():.4f}', end='')
            train_loss_list.append([epoch, train_loss / len(train_loader)])
            
            if epoch % self.conf.sswm_phase1.lr_step == 0:
                scheduler.step()
            
            if epoch % self.conf.sswm_phase1.val_interval == 0:
                self.eval()
                val_loss, hit_codes, xy_true, xy_pred, energy = 0., [], [], [], []
                for i, (X, Y) in enumerate(val_loader):
                    x_logits, z_e_x, z_q_x, x_code_idx = self(X, test=True)
                    
                    loss_recons = F.cross_entropy(torch.concat([x_logits[:,0], x_logits[:,1]], dim=1), torch.concat([X[:,0], X[:,1]], dim=1))
                    val_loss += loss_recons.item()
                    
                    # GT
                    X_true = torch.argmax(X, dim=-1)
                    
                    # Greedy Sample
                    x_logits = F.softmax(x_logits, dim=-1)
                    X_sample = torch.argmax(x_logits, dim=-1)
                    
                    xy_true.extend(X_true.cpu().detach().numpy().tolist())
                    xy_pred.extend(X_sample.cpu().detach().numpy().tolist())
                    code_idx = self.encode(Y)
                    hit_codes.extend(code_idx.view(-1).cpu().detach().numpy().tolist())
                    energy.extend(self.Energy(X).squeeze().cpu().detach().numpy().tolist())
                
                # hit num
                hit_num = np.unique(hit_codes).size
                hit_num_list.append(hit_num)
                
                print(f'\nEpoch[{epoch}/{self.conf.sswm_phase1.max_epoch}] val loss={val_loss/len(val_loader):.4f} | hit num={hit_num}/{self.conf.PESLA_SSWM.K}')
                val_loss_list.append([epoch, val_loss/len(val_loader)])
                
                v_min, v_max = X.min().item(), X.max().item()
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(np.array(xy_true)[:,0], np.array(xy_pred)[:,0], s=5)
                plt.plot([v_min, v_max], [v_min, v_max], 'r--')
                plt.xlabel('locus1_true')
                plt.ylabel('locus1_pred')
                plt.subplot(1, 2, 2)
                plt.scatter(np.array(xy_true)[:,1], np.array(xy_pred)[:,1], s=5)
                plt.plot([v_min, v_max], [v_min, v_max], 'r--')
                plt.xlabel('locus2_true')
                plt.ylabel('locus2_pred')
                plt.tight_layout()
                plt.savefig(self.conf.log_dir+f'val/phase1/regression_{epoch}.png', dpi=300)
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(np.array(xy_true)[:,0], np.array(xy_true)[:,1], s=5)
                plt.xlabel('locus1')
                plt.ylabel('locus2')
                plt.xlim(-1, self.conf.SSWM.states)
                plt.ylim(-1, self.conf.SSWM.states)
                plt.title('True')
                plt.subplot(1, 2, 2)
                plt.scatter(np.array(xy_pred)[:,0], np.array(xy_pred)[:,1], s=5)
                plt.xlabel('locus1')
                plt.ylabel('locus2')
                plt.xlim(-1, self.conf.SSWM.states)
                plt.ylim(-1, self.conf.SSWM.states)
                plt.title('Pred')
                plt.tight_layout()
                plt.savefig(self.conf.log_dir+f'val/phase1/landscape_{epoch}.png', dpi=300)
                
                plt.figure(figsize=(6,5))
                fig = plt.tricontourf(np.array(xy_pred)[:,0], np.array(xy_pred)[:,1], np.array(energy), levels=100, cmap='jet')
                plt.scatter(np.array(xy_pred)[:,0], np.array(xy_pred)[:,1], c='gray', s=5)
                plt.xlim(-1, self.conf.SSWM.states)
                plt.ylim(-1, self.conf.SSWM.states)
                plt.title('Energy Landscape (Code Space)')
                plt.colorbar(fig)
                plt.tight_layout()
                plt.savefig(self.conf.log_dir+f'val/phase1/landscape_energy_{epoch}.png', dpi=300)
                
                plt.figure(figsize=(6,5))
                fig = plt.tricontourf(np.array(xy_true)[:,0], np.array(xy_true)[:,1], np.array(energy), levels=100, cmap='jet')
                plt.xlim(0, self.conf.SSWM.states-1)
                plt.ylim(0, self.conf.SSWM.states-1)
                plt.title('Energy Landscape (Data Space)')
                plt.colorbar(fig)
                plt.tight_layout()
                plt.savefig(self.conf.log_dir+f'val/phase1/true_energy_{epoch}.png', dpi=300)
                
                xy = np.arange(0, self.conf.SSWM.states)
                sample_coords = np.dstack(np.meshgrid(xy, xy)).reshape(-1, 2)
                sample_coords_onehot = np.eye(self.conf.SSWM.states)[sample_coords.reshape(-1, 2)] # (N, loci, N_states)
                code_idxs = self.encode(torch.tensor(sample_coords_onehot, dtype=torch.float32).to(self.conf.device)).view(-1).cpu().detach().numpy()
                hit_code_num = np.unique(code_idxs).size
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
                markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
                plt.figure(figsize=(5,5))
                for i in range(len(sample_coords)):
                    color_index = code_idxs[i] % 20
                    marker_index = code_idxs[i] // np.ceil(self.conf.PESLA_SSWM.K / len(markers)).astype(int)
                    plt.scatter(sample_coords[i, 0], sample_coords[i, 1], color=colors[color_index], marker=markers[marker_index])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(-1, self.conf.SSWM.states)
                plt.ylim(-1, self.conf.SSWM.states)
                plt.title(f'Codebook | Usage Num={hit_code_num}/{self.conf.PESLA_SSWM.K}')
                plt.tight_layout()
                plt.savefig(self.conf.log_dir+f'val/phase1/codebook_{epoch}.png', dpi=300)
        
        # Save Phase1 model
        torch.save(self.state_dict(), self.conf.log_dir+'phase1.pt')
        
        # Draw loss curve
        train_loss_list = np.array(train_loss_list)
        val_loss_list = np.array(val_loss_list)
        plt.figure(figsize=(6, 7))
        plt.subplot(2, 1, 1)
        # plt.plot(train_loss_list[:, 0], train_loss_list[:, 1], label='train')
        plt.plot(val_loss_list[:, 0], val_loss_list[:, 1], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(hit_num_list)+1), hit_num_list)
        plt.xlabel('Epoch')
        plt.ylabel(f'Hit Num (K={self.conf.PESLA_SSWM.K})')
        plt.tight_layout()
        plt.savefig(self.conf.log_dir+'loss_phase1.png', dpi=300)
    
    
    def fit_phase2(self, train_loader, val_loader):
        if os.path.exists(self.conf.log_dir+'phase2.pickle'):
            with open(self.conf.log_dir+'phase2.pickle', 'rb') as f:
                [self.dynamics, self.codebook, self.energyNet] = pickle.load(f)
                self.dynamics.pe = self.dynamics.pe.to(self.conf.device)
                self.dynamics.node_energy = self.dynamics.node_energy.to(self.conf.device)
                self.dynamics.A = self.dynamics.A.to(self.conf.device)
                self.dynamics.odenet.E = self.dynamics.odenet.E.to(self.conf.device)
            print('Load pre-trained model (Phase2) ...')
            return
        else:
            os.makedirs(self.conf.log_dir+'val/phase2/', exist_ok=True)
            print(f'Training {self.conf.model} model (Phase2) ...')
        
        hit_code_idx = []
        for loader in [train_loader, val_loader]:
            for i, (X, Y) in enumerate(loader):
                _, _, _, x_code_idx = self(X)
                hit_code_idx.extend(x_code_idx.view(-1).cpu().detach().numpy().tolist())
        hit_code_idx = np.unique(hit_code_idx)
        print(f'Hit Code Index: {len(hit_code_idx)}/{self.conf.PESLA_SSWM.K}')
        
        self.codebook.embedding.weight.data = self.codebook.embedding.weight.data[hit_code_idx]
        self.codebook.K = len(hit_code_idx)
        
        node_features = self.codebook.embedding.weight.data.detach().cpu().numpy()
        node_energy = self.energyNet(self.codebook.embedding.weight.data).detach()
        self.dynamics = EnergyDynamicsSSWM(self.conf, node_features, node_energy).to(self.conf.device)
        
        os.makedirs(self.conf.log_dir+'val/phase2/', exist_ok=True)
        print(f'Training {self.conf.model} model (Phase2) ...')
        
        optimizer = torch.optim.Adam(
            list(self.dynamics.parameters()) + list(self.energyNet.parameters()), 
            lr=self.conf.sswm_phase2.lr
        )
        scheduler = ExponentialLR(optimizer, gamma=self.conf.sswm_phase2.lr_decay)
        
        train_loss_list, val_loss_list = [], []
        for epoch in range(1, self.conf.sswm_phase2.max_epoch+1):
            
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
            print(f'\rEpoch[{epoch}/{self.conf.sswm_phase2.max_epoch}] loss_h={loss_h.item():.4f} loss_p={loss_p.item():.4f} loss_l={loss_l.item():.4f}', end='')
            
            train_loss_list.append([epoch, train_loss / len(train_loader)])
            
            if epoch % self.conf.sswm_phase2.lr_step == 0:
                scheduler.step()
                
            if epoch % self.conf.sswm_phase2.val_interval == 0:
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
                print(f'\nEpoch[{epoch}/{self.conf.sswm_phase2.max_epoch}] val loss_h={loss_h.item():.4f} val loss_p={loss_p.item():.4f}')
                val_loss_list.append([epoch, val_loss/len(val_loader)])
                
                # Visualize the graph topology
                G = nx.from_numpy_array(self.dynamics.A.cpu().detach().numpy())
                pos = {i: self.dynamics.node_features_pca[i, :2] for i in range(self.dynamics.N)}
                plt.figure(figsize=(8, 8))
                plt.title(f'Source {self.encode(X[:1])[0].item()}')
                p = torch.softmax(P_logit_hat[0], 0).cpu().detach().numpy()
                nx.draw(G, pos, with_labels=True, labels={i: f'{p[i]:.2f}' for i in range(self.dynamics.N)}, node_size=300, node_color=p, font_size=8, font_color='black', edge_color='gray')
                plt.tight_layout()
                plt.savefig(self.conf.log_dir+f'val/phase2/p_{epoch}.png', dpi=300)

        # Save Phase2 model
        with open(self.conf.log_dir+'phase2.pickle', 'wb') as f:
            # pickle.dump([self.dynamics, self.codebook], f)
            pickle.dump([self.dynamics, self.codebook, self.energyNet], f)
            
        # Draw loss curve
        train_loss_list = np.array(train_loss_list)
        val_loss_list = np.array(val_loss_list)
        plt.figure(figsize=(6, 3.5))
        # plt.plot(train_loss_list[:, 0], train_loss_list[:, 1], label='train')
        plt.plot(val_loss_list[:, 0], val_loss_list[:, 1], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(self.conf.log_dir+'loss_phase2.png', dpi=300)
    
    
    def test(self):        
        os.makedirs(self.conf.log_dir+'test/', exist_ok=True)
        self.eval()
        self.to(self.conf.device)
        
        traj_num = 1000
        steps = 100
        test_sswm(self, traj_num, steps, lag=self.conf.SSWM.lag)
        