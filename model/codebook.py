import torch
import torch.nn as nn


def temperature_softmax(logits, temperature=1.0):
    return torch.softmax(logits / temperature, dim=-1)
                

class CodeBook(nn.Module):
    def __init__(self, K, D, temperature=0.01, init='normal'):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.temperature = temperature
        if init == 'normal':
            self.embedding.weight.data.normal_(0, 1)
        elif init == 'uniform':
            self.embedding.weight.data.uniform_(-1/K, 1/K)

    def forward(self, z_e_x, greedy=True):
        # z_e_x shape: (batch, D)

        with torch.no_grad():
            B, K = z_e_x.size(0), self.embedding.weight.size(0)
            codebook_vectors = self.embedding.weight.unsqueeze(0)  # codebook shape: (1, K, D)
            z_e_x_ = z_e_x.unsqueeze(1) # (batch, 1, D)
            codebook_vectors = codebook_vectors.repeat(B, 1, 1)  # codebook shape: (batch, K, D)
            z_e_x_ = z_e_x_.repeat(1, K, 1) # z_e_x shape: (batch, K, D)
        
            distances = torch.sum((z_e_x_ - codebook_vectors) ** 2, dim=-1)  # distances shape: (batch, K)
            
            if greedy:
                index = torch.argmin(distances, dim=-1)  # min_distances shape: (batch)
            else:
                probs = temperature_softmax(-distances, temperature=self.temperature)
                index = torch.multinomial(probs, 1).squeeze()
        
        return index
    
    def straight_through(self, z_e_x, greedy=True):
        # z_e_x shape: (batch, D)

        index = self.forward(z_e_x, greedy)     
        z_q_x = self.embedding(index)  # z_q_x shape: (batch, D)
        
        # straight-through gradient
        sg_z_e_x = z_e_x + (z_q_x - z_e_x).detach()
        
        return sg_z_e_x, z_q_x, index
    
    def lookup(self, index):
        return self.embedding(index)