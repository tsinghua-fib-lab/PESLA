import math
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from fast_pagerank import pagerank # ref: fast-pagerank
from torchdiffeq import odeint




class EnergyPredictor(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, 1),
            nn.Softplus(), # limit the output to [0, +inf]
        )
    
    def forward(self, code):
        # code: (B, D)
        return torch.log(self.energy_net(code)) # (B, 1)
    
    
class GCN(nn.Module):
    def __init__(self, in_dim, nhidden, out_dim, adj, positive=False):
        super(GCN, self).__init__()
        self.positive = positive
        self.w1 = nn.Linear(in_dim, nhidden, bias=True)
        self.w2 = nn.Linear(nhidden, out_dim, bias=True)
        
        self.adj = self.normalize_adj(adj)
        
    def normalize_adj(self, adj):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        D = torch.pow(adj.sum(1).float(), -0.5)
        D = torch.diag(D).to(adj.device)
        return D @ adj @ D

    def forward(self, x):
        self.adj = self.adj.to(x.device)
        
        h1 = torch.relu(self.w1(self.adj @ x))
        h2 = self.w2(self.adj @ h1)
        return torch.relu(h2) if self.positive else h2


class log_Graph_Fokker_Planck_ODEFunc(nn.Module):
    def __init__(self, E, A, nhidden, pe_dim=3, k=10.):
        super().__init__()
        
        self.k = k
        self.E = E.squeeze() # (N,)
        self.A = A # (N, N)
        
        self.w_k = nn.Linear(pe_dim, nhidden, bias=True)
        self.w_q = nn.Linear(pe_dim, nhidden, bias=True)
        self.beta = nn.Parameter(torch.rand(nhidden), requires_grad=True) # (D,)
        
        self.pe = None
        
    def forward(self, t, h):
        """ Fokker-Planck on Graph ODE function in log space """
        self.A = self.A.to(h.device)
        
        # log_h: (B, N, D)
        B, N, D = h.size()
        log_h = torch.log(h + 1e-8) # (B, N, D)
        
        # W
        K = self.w_k(self.pe) # (B, N, D)
        Q = self.w_q(self.pe) # (B, N, D)
        W = torch.softmax(self.A.unsqueeze(0) * torch.bmm(K, Q.transpose(1, 2)) / math.sqrt(D), dim=-1) # (B, N, N)
        W = W.unsqueeze(-1).repeat(1, 1, 1, D) # (B, N, N, D)
        
        # Drift term
        E_diff = self.E.unsqueeze(0) - self.E.unsqueeze(1) # (N, N)
        batch_E_diff = E_diff.unsqueeze(0).repeat(B, 1, 1) # (B, N, N)
        log_h_diff = log_h.unsqueeze(1) - log_h.unsqueeze(2) # (B, N, N, D)
        beta = self.beta.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, N, N, 1) # (B, N, N, D)
        drift_term = W * (batch_E_diff.unsqueeze(-1) + beta * log_h_diff) # (B, N, N, D)
        
        # soften sign function by sigmoid
        sign_matrix = torch.sigmoid(self.k*E_diff) # (N, N)
        
        weighted_h = sign_matrix.unsqueeze(0).unsqueeze(-1) * h.unsqueeze(1) + (1 - sign_matrix.unsqueeze(0).unsqueeze(-1)) * h.unsqueeze(2) # (B, N, N, D)
        
        dh_dt = torch.sum(drift_term * weighted_h, dim=-2) # (B, N, D)
        
        return dh_dt



class EnergyDynamics4Well(nn.Module):
    def __init__(self, conf, node_features, node_energy):
        super().__init__()
        
        self.conf = conf
        self.N = len(node_features)
        self.node_energy = node_energy # (N, 1)
        self.A = self.BuildVoronoiGraph(node_features) # (N, N)
        self.pe = self.positional_encoding(self.A.cpu().detach().numpy()) # (N, N, len(scales))
        self.odenet = log_Graph_Fokker_Planck_ODEFunc(self.node_energy, self.A, conf.PESLA_4_Well.nhidden, pe_dim=self.pe.size(-1))
        
        self.encoder = GCN(3+1+1, conf.PESLA_4_Well.nhidden, conf.PESLA_4_Well.nhidden, self.A, positive=True)
        self.decoder = GCN(conf.PESLA_4_Well.nhidden, conf.PESLA_4_Well.nhidden, 1, self.A)
        
        self.apply(self.weights_init)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.constant_(m.bias, 0)
               
    def BuildVoronoiGraph(self, node_features):

        adj = torch.zeros(self.N, self.N)
        
        # Dim-reduce node features for Delaunay triangulation
        pca = PCA(n_components=2)
        self.node_features_pca = pca.fit_transform(node_features)
        print(f'PCA explained variance ratio: {pca.explained_variance_ratio_.round(3)} sum: {pca.explained_variance_ratio_.sum():.3f}')
        
        # Create a one-hop topology graph (Voronoi diagram) by Delaunay triangulation
        tri = Delaunay(self.node_features_pca)
        adj[tri.simplices[:, 0], tri.simplices[:, 1]] = 1.
        adj[tri.simplices[:, 1], tri.simplices[:, 2]] = 1.
        adj[tri.simplices[:, 2], tri.simplices[:, 0]] = 1.
        print(f'Voronoi graph: {adj.sum().item()} edges')
        print(f'Average degree: {adj.sum().item() / self.N:.2f}')
        
        # Visualize the graph topology
        G = nx.from_numpy_array(adj.numpy())
        pos = {i: self.node_features_pca[i, :2] for i in range(self.N)}
        plt.figure(figsize=(6, 6))
        # nx.draw(G, pos, with_labels=True, node_size=300, node_color=pe.mean(-1), cmap=plt.cm.Blues, font_size=8, font_color='black', edge_color='gray')
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=8, font_color='black', edge_color='gray')
        plt.savefig(self.conf.log_dir+'init_voronoi.png', dpi=300)
                
        return adj.to(self.conf.device)
    
    def positional_encoding(self, adj, scales=[0.999, 0.99, 0.9]):
        # Personalized PageRank for each source-target pair, adj: (N, N)
        # Returen: (i, j, k) is the PageRank score of source i to target j with scale k
        pe = np.zeros((self.N, self.N, len(scales)))
        for source_idx in range(self.N):
            personalize = np.zeros(self.N)
            personalize[source_idx] = 1.0
            for i, p in enumerate(scales):
                scores = pagerank(adj, p=p, personalize=personalize)
                pe[source_idx, :, i] = scores / scores.max()
        
        return torch.from_numpy(pe).float().to(self.conf.device) # (N, N, len(scales))
    
    def build_node_features(self, x_code_idx):
        # x_code_idx:(B,)
        # Initial distribution
        p0 = torch.zeros(len(x_code_idx), self.N, 1).to(x_code_idx.device) # (B, N, 1)
        p0[torch.arange(len(x_code_idx)), x_code_idx, 0] = 1.0
        
        # Build node features
        pe = self.pe[x_code_idx] # (B, N, len(scales))
        batch_node_energy = self.node_energy.unsqueeze(0).repeat(len(x_code_idx), 1, 1) # (B, N, 1)
        node_features = torch.cat([pe, batch_node_energy, p0], dim=-1) # (B, N, len(scales)+2)
        
        return node_features
    
    def H(self, x_code_idx):
        # Build node features
        node_features = self.build_node_features(x_code_idx)
        # Encode from p to h
        h = self.encoder(node_features) # (B, N, nhidden)
        return h
    
    def evolve_one_step(self, x_code_idx, lag=1):
        # x_code_idx:(B,)
        # Initial distribution
        p0 = torch.zeros(len(x_code_idx), self.N, 1).to(x_code_idx.device) # (B, N, 1)
        p0[torch.arange(len(x_code_idx)), x_code_idx, 0] = 1.0
        
        # Build node features
        pe = self.pe[x_code_idx] # (B, N, len(scales))
        batch_node_energy = self.node_energy.unsqueeze(0).repeat(len(x_code_idx), 1, 1) # (B, N, 1)
        node_features = torch.cat([pe, batch_node_energy, p0], dim=-1) # (B, N, len(scales)+2)
        
        # Encode from p0 to h0
        h0 = self.encoder(node_features) # (B, N, nhidden)
        
        # Solve ODE in latent space
        tspan = torch.linspace(0, 1e-2*lag, 2).to(x_code_idx.device)
        self.odenet.pe = pe
        h1 = odeint(self.odenet, h0, tspan, method='euler')[-1] # (B, N, nhidden)
        
        # Decode from h to p
        p1_logit = self.decoder(h1).squeeze() # (B, N)
        
        return h1, p1_logit





class EnergyDynamicsProtein(nn.Module):
    def __init__(self, conf, node_features, node_energy):
        super().__init__()
        
        self.conf = conf
        self.N = len(node_features)
        self.node_energy = node_energy # (N, 1)
        self.A = self.BuildVoronoiGraph(node_features) # (N, N)
        self.pe = self.positional_encoding(self.A.cpu().detach().numpy()) # (N, N, len(scales))
        self.odenet = log_Graph_Fokker_Planck_ODEFunc(self.node_energy, self.A, conf.PESLA_Homeodomain.nhidden, pe_dim=self.pe.size(-1))
        
        self.encoder = GCN(3+1+1, conf.PESLA_Homeodomain.nhidden, conf.PESLA_Homeodomain.nhidden, self.A, positive=True)
        self.decoder = GCN(conf.PESLA_Homeodomain.nhidden, conf.PESLA_Homeodomain.nhidden, 1, self.A)
        
        self.apply(self.weights_init)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.constant_(m.bias, 0)
               
    def BuildVoronoiGraph(self, node_features):

        adj = torch.zeros(self.N, self.N)
        
        # Dim-reduce node features for Delaunay triangulation
        pca = PCA(n_components=3)
        self.node_features_pca = pca.fit_transform(node_features)
        print(f'PCA explained variance ratio: {pca.explained_variance_ratio_.round(3)} sum: {pca.explained_variance_ratio_.sum():.3f}')
        
        # Create a one-hop topology graph (Voronoi diagram) by Delaunay triangulation
        tri = Delaunay(self.node_features_pca)
        adj[tri.simplices[:, 0], tri.simplices[:, 1]] = 1.
        adj[tri.simplices[:, 1], tri.simplices[:, 2]] = 1.
        adj[tri.simplices[:, 2], tri.simplices[:, 0]] = 1.
        print(f'Voronoi graph: {adj.sum().item()} edges')
        print(f'Average degree: {adj.sum().item() / self.N:.2f}')
        
        # Visualize the graph topology
        G = nx.from_numpy_array(adj.numpy())
        pos = {i: self.node_features_pca[i, :2] for i in range(self.N)}
        plt.figure(figsize=(6, 6))
        # nx.draw(G, pos, with_labels=True, node_size=300, node_color=pe.mean(-1), cmap=plt.cm.Blues, font_size=8, font_color='black', edge_color='gray')
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=8, font_color='black', edge_color='gray')
        plt.savefig(self.conf.log_dir+'init_voronoi.png', dpi=300)
                
        return adj.to(self.conf.device)
    
    def positional_encoding(self, adj, scales=[0.999, 0.99, 0.9]):
        # Personalized PageRank for each source-target pair, adj: (N, N)
        # Returen: (i, j, k) is the PageRank score of source i to target j with scale k
        pe = np.zeros((self.N, self.N, len(scales)))
        for source_idx in range(self.N):
            personalize = np.zeros(self.N)
            personalize[source_idx] = 1.0
            for i, p in enumerate(scales):
                scores = pagerank(adj, p=p, personalize=personalize)
                pe[source_idx, :, i] = scores / scores.max()
        
        return torch.from_numpy(pe).float().to(self.conf.device) # (N, N, len(scales))
    
    def build_node_features(self, x_code_idx):
        # x_code_idx:(B,)
        # Initial distribution
        p0 = torch.zeros(len(x_code_idx), self.N, 1).to(x_code_idx.device) # (B, N, 1)
        p0[torch.arange(len(x_code_idx)), x_code_idx, 0] = 1.0
        
        # Build node features
        pe = self.pe[x_code_idx] # (B, N, len(scales))
        batch_node_energy = self.node_energy.unsqueeze(0).expand(len(x_code_idx), -1, -1) # (B, N, 1)
        node_features = torch.cat([pe, batch_node_energy, p0], dim=-1) # (B, N, len(scales)+2)
        
        return node_features
    
    def H(self, x_code_idx):
        # Build node features
        node_features = self.build_node_features(x_code_idx)
        # Encode from p to h
        h = self.encoder(node_features) # (B, N, nhidden)
        return h
    
    def evolve_one_step(self, x_code_idx, lag=1):
        # x_code_idx:(B,)
        # Initial distribution
        p0 = torch.zeros(len(x_code_idx), self.N, 1).to(x_code_idx.device) # (B, N, 1)
        p0[torch.arange(len(x_code_idx)), x_code_idx, 0] = 1.0
        
        # Build node features
        pe = self.pe[x_code_idx] # (B, N, len(scales))
        batch_node_energy = self.node_energy.unsqueeze(0).expand(len(x_code_idx), -1, -1) # (B, N, 1)
        node_features = torch.cat([pe, batch_node_energy, p0], dim=-1) # (B, N, len(scales)+2)
        
        # Encode from p0 to h0
        h0 = self.encoder(node_features) # (B, N, nhidden)
        
        # Solve ODE in latent space
        tspan = torch.linspace(0, 1e-2*lag, 2).to(x_code_idx.device)
        self.odenet.pe = pe
        h1 = odeint(self.odenet, h0, tspan, method='euler')[-1] # (B, N, nhidden)
        
        # Decode from h to p
        p1_logit = self.decoder(h1).squeeze() # (B, N)
        
        return h1, p1_logit


class EnergyDynamicsSSWM(nn.Module):
    def __init__(self, conf, node_features, node_energy):
        super().__init__()
        
        self.conf = conf
        self.N = len(node_features)
        self.node_energy = node_energy # (N, 1)
        self.A = self.BuildVoronoiGraph(node_features) # (N, N)
        self.pe = self.positional_encoding(self.A.cpu().detach().numpy()) # (N, N, len(scales))
        self.odenet = log_Graph_Fokker_Planck_ODEFunc(self.node_energy, self.A, conf.EnergyLandscape_SSWM.nhidden, pe_dim=self.pe.size(-1))
        
        self.encoder = GCN(3+1+1, conf.EnergyLandscape_SSWM.nhidden, conf.EnergyLandscape_SSWM.nhidden, self.A, positive=True)
        self.decoder = GCN(conf.EnergyLandscape_SSWM.nhidden, conf.EnergyLandscape_SSWM.nhidden, 1, self.A)
        
        self.apply(self.weights_init)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.constant_(m.bias, 0)
               
    def BuildVoronoiGraph(self, node_features):

        adj = torch.zeros(self.N, self.N)
        
        # Dim-reduce node features for Delaunay triangulation
        pca = PCA(n_components=5)
        self.node_features_pca = pca.fit_transform(node_features)
        print(f'PCA explained variance ratio: {pca.explained_variance_ratio_.round(3)} sum: {pca.explained_variance_ratio_.sum():.3f}')
        
        # Create a one-hop topology graph (Voronoi diagram) by Delaunay triangulation
        tri = Delaunay(self.node_features_pca)
        adj[tri.simplices[:, 0], tri.simplices[:, 1]] = 1.
        adj[tri.simplices[:, 1], tri.simplices[:, 2]] = 1.
        adj[tri.simplices[:, 2], tri.simplices[:, 0]] = 1.
        print(f'Voronoi graph: {adj.sum().item()} edges')
        print(f'Average degree: {adj.sum().item() / self.N:.2f}')
                
        # Visualize the graph topology
        G = nx.from_numpy_array(adj.numpy())
        pos = {i: self.node_features_pca[i, :2] for i in range(self.N)}
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=8, font_color='black', edge_color='gray')
        plt.savefig(self.conf.log_dir+'init_voronoi.png', dpi=300)
                
        return adj.to(self.conf.device)
    
    def positional_encoding(self, adj, scales=[0.999, 0.99, 0.9]):
        # Personalized PageRank for each source-target pair, adj: (N, N)
        # Returen: (i, j, k) is the PageRank score of source i to target j with scale k
        pe = np.zeros((self.N, self.N, len(scales)))
        for source_idx in range(self.N):
            personalize = np.zeros(self.N)
            personalize[source_idx] = 1.0
            for i, p in enumerate(scales):
                scores = pagerank(adj, p=p, personalize=personalize)
                pe[source_idx, :, i] = scores / scores.max()
        
        return torch.from_numpy(pe).float().to(self.conf.device) # (N, N, len(scales))
    
    def build_node_features(self, x_code_idx):
        # x_code_idx:(B,)
        # Initial distribution
        p0 = torch.zeros(len(x_code_idx), self.N, 1).to(x_code_idx.device) # (B, N, 1)
        p0[torch.arange(len(x_code_idx)), x_code_idx, 0] = 1.0
        
        # Build node features
        pe = self.pe[x_code_idx] # (B, N, len(scales))
        batch_node_energy = self.node_energy.unsqueeze(0).expand(len(x_code_idx), -1, -1) # (B, N, 1)
        node_features = torch.cat([pe, batch_node_energy, p0], dim=-1) # (B, N, len(scales)+2)
        
        return node_features
    
    def H(self, x_code_idx):
        # Build node features
        node_features = self.build_node_features(x_code_idx)
        # Encode from p to h
        h = self.encoder(node_features) # (B, N, nhidden)
        return h
    
    def evolve_one_step(self, x_code_idx, lag=1):
        # x_code_idx:(B,)
        # Initial distribution
        p0 = torch.zeros(len(x_code_idx), self.N, 1).to(x_code_idx.device) # (B, N, 1)
        p0[torch.arange(len(x_code_idx)), x_code_idx, 0] = 1.0
        
        # Build node features
        pe = self.pe[x_code_idx] # (B, N, len(scales))
        batch_node_energy = self.node_energy.unsqueeze(0).expand(len(x_code_idx), -1, -1) # (B, N, 1)
        node_features = torch.cat([pe, batch_node_energy, p0], dim=-1) # (B, N, len(scales)+2)
        
        # Encode from p0 to h0
        h0 = self.encoder(node_features) # (B, N, nhidden)
        
        # Solve ODE in latent space
        tspan = torch.linspace(0, 1e-2*lag, 2).to(x_code_idx.device)
        self.odenet.pe = pe
        h1 = odeint(self.odenet, h0, tspan, method='euler')[-1] # (B, N, nhidden)
        
        # Decode from h to p
        p1_logit = self.decoder(h1).squeeze() # (B, N)
        
        return h1, p1_logit