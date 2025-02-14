import torch
import torch.nn as nn



def Network4Well(input_dim, nhidden, code_dim):
    # input: (B, C), C is the number of features in the time series
    encoder = nn.Sequential(
            nn.Linear(input_dim, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, code_dim),
            nn.BatchNorm1d(code_dim),
            nn.ReLU(),
    )
    
    class GaussianDecoder(nn.Module):
        def __init__(self, input_dim, code_dim, nhidden, scale=0.05):
            super(GaussianDecoder, self).__init__()
            
            self.net = nn.Sequential(
                nn.Linear(code_dim, nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, nhidden),
            )
            
            self.decoder_mu = nn.Sequential(
                nn.Linear(nhidden, input_dim)
            )
            
            self.decoder_std  = nn.Sequential(
                nn.Linear(nhidden, input_dim),
                nn.Sigmoid()
            )
            self.scale = scale
        
        def forward(self, z):
            
            z = self.net(z)
            mu = self.decoder_mu(z)
            std = self.decoder_std(z) * self.scale
            
            eps = torch.randn_like(std)
            reconstruction = mu + eps * std
            
            return reconstruction, mu, std
    
    decoder = GaussianDecoder(input_dim, code_dim, nhidden)

    return encoder, decoder



def NetworkProtein(input_dim, nhidden, code_dim):
    # input: (B, C), C is the number of features in the time series
    
    class Encoder(nn.Module):
        def __init__(self, input_dim, code_dim, nhidden):
            super(Encoder, self).__init__()
            
            self.net = nn.Sequential(
                nn.Linear(input_dim, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, code_dim),
                nn.BatchNorm1d(code_dim),
                nn.ReLU(),
            )
            self.K = 100
        
        def forward(self, x):
            return self.net(x) / self.K
    
    class GaussianDecoder(nn.Module):
        def __init__(self, input_dim, code_dim, nhidden, scale=0.05):
            super(GaussianDecoder, self).__init__()
            
            self.net = nn.Sequential(
                nn.Linear(code_dim, nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, nhidden),
            )
            
            self.decoder_mu = nn.Sequential(
                nn.Linear(nhidden, input_dim)
            )
            
            self.decoder_std  = nn.Sequential(
                nn.Linear(nhidden, input_dim),
                nn.Sigmoid()
            )
            self.K = 100
            self.scale = scale
        
        def forward(self, z):
            
            z *= self.K
            z = self.net(z)
            mu = self.decoder_mu(z)
            std = self.decoder_std(z) * self.scale
            
            eps = torch.randn_like(std)
            reconstruction = mu + eps * std
            
            return reconstruction, mu, std
    
    encoder =  Encoder(input_dim, code_dim, nhidden)
    decoder = GaussianDecoder(input_dim, code_dim, nhidden)

    return encoder, decoder



def NetworkSSWM(n_loci, nhidden, code_dim, n_states, K):
    # input: (B, C), C is the number of loci
    class MutliLociEncoder(nn.Module):
        def __init__(self, n_loci, code_dim, nhidden, n_states, K):
            super().__init__()
            assert code_dim % n_loci == 0, 'code_dim should be divisible by n_loci'
            
            self.K = K
            self.encoders = nn.ModuleList([])
            out_dim = code_dim // n_loci
            for _ in range(n_loci):
                self.encoders.append(nn.Sequential(
                    nn.Linear(n_states, nhidden),
                    nn.Tanh(),
                    nn.BatchNorm1d(nhidden),
                    nn.Linear(nhidden, out_dim),
               ))
        
        def forward(self, x):
            # x: (B, n_loci, n_states)
            z = []
            for i in range(n_loci):
                z.append(self.encoders[i](x[:, i, :])) # (B, out_dim)
            concat_z = torch.concat(z, dim=1) # (B, code_dim)
            return concat_z / self.K
    
    class MutliLociDecoder(nn.Module):
        def __init__(self, n_loci, code_dim, nhidden, n_states, K):
            super().__init__()
            assert code_dim % n_loci == 0, 'code_dim should be divisible by n_loci'
            
            self.K = K
            self.decoders = nn.ModuleList([])
            self.in_dim = code_dim // n_loci
            for _ in range(n_loci):
                self.decoders.append(nn.Sequential(
                    nn.Linear(self.in_dim, nhidden),
                    nn.ReLU(),
                    nn.Linear(nhidden, n_states),
               ))
        
        def forward(self, z):
            # z: (B, code_dim)
            z *= self.K
            logits = []
            for i in range(n_loci):
                logits.append(self.decoders[i](z[:, i*self.in_dim:(i+1)*self.in_dim])) # (B, n_states)
            return torch.stack(logits, dim=1) # (B, n_loci, n_states)
        
    encoder = MutliLociEncoder(n_loci, code_dim, nhidden, n_states, K)
    decoder = MutliLociDecoder(n_loci, code_dim, nhidden, n_states, K)
    
    return encoder, decoder




def AE(conf):
    if conf.system == '_4_Well':
        feature_dim, nhidden, code_dim = conf['_4_Well'].feature_dim, conf.PESLA_4_Well.nhidden, conf.PESLA_4_Well.code_dim
        return Network4Well(input_dim=feature_dim, nhidden=nhidden, code_dim=code_dim)
    elif conf.system == 'Homeodomain':
        feature_dim, nhidden, code_dim = conf['Homeodomain'].feature_dim, conf.PESLA_Homeodomain.nhidden, conf.PESLA_Homeodomain.code_dim
        return NetworkProtein(input_dim=feature_dim, nhidden=nhidden, code_dim=code_dim)
    elif conf.system == 'SSWM':
        loci, states, nhidden, code_dim = conf['SSWM'].loci, conf['SSWM'].states, conf.PESLA_SSWM.nhidden, conf.PESLA_SSWM.code_dim
        return NetworkSSWM(n_loci=loci, nhidden=nhidden, code_dim=code_dim, n_states=states, K=conf.PESLA_SSWM.K)
    else:
        raise ValueError(f'Invalid system: {conf.system}')