import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



class Dataset_4_Well(Dataset):
    
    def __init__(self, conf, mode='phase1_train', traj=None):
        super().__init__()
        self.conf = conf
        self.mode = mode

        # Preprocess data
        try:
            if 'phase1' in self.mode:
                processed_data = np.load(conf.data_dir+f'{mode}.npz')
            elif 'phase2' in self.mode:
                lag = self.conf[self.conf.system].lag
                processed_data = np.load(conf.data_dir+f'{mode}_lag{lag}.npz')
            self.X = torch.from_numpy(processed_data['X']).float().to(self.conf.device)
            self.Y = torch.from_numpy(processed_data['Y']).float().to(self.conf.device)
            self.mean_y, self.std_y = processed_data['mean_y'], processed_data['std_y']
        except:
            self.process(traj)
        
        # Data ratio
        random_idx = np.random.choice(np.arange(len(self.X)), int(len(self.X)*conf.data_ratio), replace=False)
        self.X, self.Y = self.X[random_idx], self.Y[random_idx]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
    
    def process(self, traj=None):
        # origin data: (traj_num, N, feature_dim)
        if traj is None:
            traj = np.load(self.conf.data_dir+'traj.npz')['traj']
        
        # Lag
        if 'phase2' in self.mode:
            lag = self.conf[self.conf.system].lag
            traj = traj[:, ::lag]
        
        # Sliding window
        X = traj[:, :-1]
        Y = traj[:, 1:]
        
        # Normalize
        if self.conf.data_norm:
            self.mean_data = X.mean(axis=(1,), keepdims=True)
            self.std_data = X.std(axis=(1,), keepdims=True)
            X = (X - self.mean_data) / self.std_data
            Y = (Y - self.mean_data) / self.std_data
        else:
            self.mean_data, self.std_data = 0, 1
            
        # Flatten
        X = X.reshape(-1, X.shape[-1])
        Y = Y.reshape(-1, X.shape[-1])
        
        # Split train and val
        train_size = int(X.shape[0] * self.conf.train_ratio)
        val_size = X.shape[0] - train_size
        train_idx = np.random.choice(np.arange(X.shape[0]), train_size, replace=False)
        val_idx = np.random.choice(np.setdiff1d(np.arange(X.shape[0]), train_idx), val_size, replace=False)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        # Save        
        if 'train' in self.mode:
            self.X, self.Y = X_train, Y_train
        elif 'val' in self.mode:
            self.X, self.Y = X_val, Y_val
        
        if 'phase1' in self.mode:
            np.savez(self.conf.data_dir+f'{self.mode}.npz', X=self.X, Y=self.Y, mean_y=self.mean_data, std_y=self.std_data)
        elif 'phase2' in self.mode:
            np.savez(self.conf.data_dir+f'{self.mode}_lag{lag}.npz', X=self.X, Y=self.Y, mean_y=self.mean_data, std_y=self.std_data)
        
        # Convert to torch tensor
        self.X = torch.from_numpy(self.X).float().to(self.conf.device) # N-1, feature_dim
        self.Y = torch.from_numpy(self.Y).float().to(self.conf.device) # N-1, feature_dim
        
        
    def getLoader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=True)



class Dataset_SSWM(Dataset):
    
    def __init__(self, conf, mode='phase1_train', traj=None):
        super().__init__()
        self.conf = conf
        self.mode = mode

        # Preprocess data
        try:
            if 'phase1' in self.mode:
                processed_data = np.load(conf.data_dir+f'{mode}.npz')
            elif 'phase2' in self.mode:
                lag = self.conf[self.conf.system].lag
                processed_data = np.load(conf.data_dir+f'{mode}_lag{lag}.npz')
            self.X = torch.from_numpy(processed_data['X']).float().to(self.conf.device)
            self.Y = torch.from_numpy(processed_data['Y']).float().to(self.conf.device)
            self.mean_y, self.std_y = processed_data['mean_y'], processed_data['std_y']
        except:
            self.process(traj)
        
        # Data ratio
        random_idx = np.random.choice(np.arange(len(self.X)), int(len(self.X)*conf.data_ratio), replace=False)
        self.X, self.Y = self.X[random_idx], self.Y[random_idx]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
    
    def process(self, traj=None):
        # origin data: (N, loci)
        if traj is None:
            traj = np.load(self.conf.data_dir+'traj.npz')['traj'] # (Traj_num, steps, loci)
        
        # Encode to one-hot
        nstates = self.conf[self.conf.system].states
        traj_onehot = np.eye(nstates)[traj.astype(int)] # (Traj_num, steps, loci, nstates)
        
        # Lag
        if 'phase2' in self.mode:
            lag = self.conf[self.conf.system].lag
            traj = traj[:, ::lag]
        
        # Sliding window
        X = traj_onehot[:, :-1]
        Y = traj_onehot[:, 1:]
        
        # Normalize
        if self.conf.data_norm:
            self.mean_data = X.mean(axis=(1,), keepdims=True)
            self.std_data = X.std(axis=(1,), keepdims=True)
            X = (X - self.mean_data) / self.std_data
            Y = (Y - self.mean_data) / self.std_data
        else:
            self.mean_data, self.std_data = 0, 1
            
        # Flatten
        X = X.reshape(-1, X.shape[-2], X.shape[-1]) # (traj_num*(steps//downsample-1), loci, nstates)
        Y = Y.reshape(-1, X.shape[-2], X.shape[-1])
        
        # Split train and val
        train_size = int(X.shape[0] * self.conf.train_ratio)
        val_size = X.shape[0] - train_size
        train_idx = np.random.choice(np.arange(X.shape[0]), train_size, replace=False)
        val_idx = np.random.choice(np.setdiff1d(np.arange(X.shape[0]), train_idx), val_size, replace=False)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        # Save
        if 'train' in self.mode:
            self.X, self.Y = X_train, Y_train
        elif 'val' in self.mode:
            self.X, self.Y = X_val, Y_val

        if 'phase1' in self.mode:
            np.savez(self.conf.data_dir+f'{self.mode}.npz', X=self.X, Y=self.Y, mean_y=self.mean_data, std_y=self.std_data)
        elif 'phase2' in self.mode:
            np.savez(self.conf.data_dir+f'{self.mode}_lag{lag}.npz', X=self.X, Y=self.Y, mean_y=self.mean_data, std_y=self.std_data)
        
        # Convert to torch tensor
        self.X = torch.from_numpy(self.X).float().to(self.conf.device) # N-1, loci, nstates
        self.Y = torch.from_numpy(self.Y).float().to(self.conf.device) # N-1, loci, nstates
        
        
    def getLoader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=True)




class Dataset_Protein(Dataset):
    
    def __init__(self, conf, mode='phase1_train', traj=None, protein=None):
        super().__init__()
        self.conf = conf
        self.mode = mode
        self.protein = protein

        # Preprocess data
        try:
            if 'phase1' in self.mode:
                processed_data = np.load(conf.data_dir+f'{mode}.npz')
            elif 'phase2' in self.mode:
                lag = self.conf[self.conf.system].lag
                processed_data = np.load(conf.data_dir+f'{mode}_lag{lag}.npz')
            self.X = torch.from_numpy(processed_data['X']).float().to(self.conf.device)
            self.Y = torch.from_numpy(processed_data['Y']).float().to(self.conf.device)
            self.mean_y, self.std_y = processed_data['mean_y'], processed_data['std_y']
            self.vmin, self.vmax = processed_data['vmin'], processed_data['vmax']
        except:
            self.process(traj)
        
        # Data ratio
        random_idx = np.random.choice(np.arange(len(self.X)), int(len(self.X)*conf.data_ratio), replace=False)
        self.X, self.Y = self.X[random_idx], self.Y[random_idx]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
    
    def process(self, traj):
        # origin data: (T, 2), T=100000, dt=0.001us
        if traj is None:
            traj = np.load(self.conf.data_dir+f'{self.conf.system}-tica-0.npy') # (steps, 2)

        # Lag
        if 'phase2' in self.mode:
            if self.protein is None:
                lag = self.conf[self.conf.system].lag
            else:
                lag = self.conf[self.protein].lag
            traj = traj[::lag]
        
        # Sliding window
        X = traj[:-1]
        Y = traj[1:]
        
        # Normalize
        if self.conf.data_norm:
            self.mean_data = X.mean(axis=(0,), keepdims=True)
            self.std_data = X.std(axis=(0,), keepdims=True)
            X = (X - self.mean_data) / (self.std_data + 1e-10)
            Y = (Y - self.mean_data) / (self.std_data + 1e-10)
        else:
            self.mean_data, self.std_data = 0, 1
        self.vmin = traj.min(axis=0, keepdims=True)
        self.vmax = traj.max(axis=0, keepdims=True)
        
        # Split train and val
        train_size = int(X.shape[0] * self.conf.train_ratio)
        val_size = X.shape[0] - train_size
        train_idx = np.random.choice(np.arange(X.shape[0]), train_size, replace=False)
        val_idx = np.random.choice(np.setdiff1d(np.arange(X.shape[0]), train_idx), val_size, replace=False)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        # Save
        if 'train' in self.mode:
            self.X, self.Y = X_train, Y_train
        elif 'val' in self.mode:
            self.X, self.Y = X_val, Y_val

        if 'phase1' in self.mode:
            np.savez(self.conf.data_dir+f'{self.mode}.npz', X=self.X, Y=self.Y, mean_y=self.mean_data, std_y=self.std_data, vmin=self.vmin, vmax=self.vmax)
        elif 'phase2' in self.mode:
            np.savez(self.conf.data_dir+f'{self.mode}_lag{lag}.npz', X=self.X, Y=self.Y, mean_y=self.mean_data, std_y=self.std_data, vmin=self.vmin, vmax=self.vmax)
        
        # Convert to torch tensor
        self.X = torch.from_numpy(self.X).float().to(self.conf.device) # T-1, 2
        self.Y = torch.from_numpy(self.Y).float().to(self.conf.device) # T-1, 2
        
        
    def getLoader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=True)



class Dataset_SSWM(Dataset):
    
    def __init__(self, conf, mode='phase1_train', traj=None):
        super().__init__()
        self.conf = conf
        self.mode = mode

        # Preprocess data
        try:
            if 'phase1' in self.mode:
                processed_data = np.load(conf.data_dir+f'{mode}.npz')
            elif 'phase2' in self.mode:
                lag = self.conf[self.conf.system].lag
                processed_data = np.load(conf.data_dir+f'{mode}_lag{lag}.npz')
            self.X = torch.from_numpy(processed_data['X']).float().to(self.conf.device)
            self.Y = torch.from_numpy(processed_data['Y']).float().to(self.conf.device)
            self.mean_y, self.std_y = processed_data['mean_y'], processed_data['std_y']
        except:
            self.process(traj)
        
        # Data ratio
        random_idx = np.random.choice(np.arange(len(self.X)), int(len(self.X)*conf.data_ratio), replace=False)
        self.X, self.Y = self.X[random_idx], self.Y[random_idx]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
    
    def process(self, traj=None):
        # origin data: (N, loci)
        if traj is None:
            traj = np.load(self.conf.data_dir+'traj.npz')['traj'] # (Traj_num, steps, loci)
        
        # Encode to one-hot
        nstates = self.conf[self.conf.system].states
        traj_onehot = np.eye(nstates)[traj.astype(int)] # (Traj_num, steps, loci, nstates)
        
        # Lag
        if 'phase2' in self.mode:
            lag = self.conf[self.conf.system].lag
            traj = traj[:, ::lag]
        
        # Sliding window
        X = traj_onehot[:, :-1]
        Y = traj_onehot[:, 1:]
        
        # Normalize
        if self.conf.data_norm:
            self.mean_data = X.mean(axis=(1,), keepdims=True)
            self.std_data = X.std(axis=(1,), keepdims=True)
            X = (X - self.mean_data) / self.std_data
            Y = (Y - self.mean_data) / self.std_data
        else:
            self.mean_data, self.std_data = 0, 1
            
        # Flatten
        X = X.reshape(-1, X.shape[-2], X.shape[-1]) # (traj_num*(steps//downsample-1), loci, nstates)
        Y = Y.reshape(-1, X.shape[-2], X.shape[-1])
        
        # Split train and val
        train_size = int(X.shape[0] * self.conf.train_ratio)
        val_size = X.shape[0] - train_size
        train_idx = np.random.choice(np.arange(X.shape[0]), train_size, replace=False)
        val_idx = np.random.choice(np.setdiff1d(np.arange(X.shape[0]), train_idx), val_size, replace=False)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        # Save
        if 'train' in self.mode:
            self.X, self.Y = X_train, Y_train
        elif 'val' in self.mode:
            self.X, self.Y = X_val, Y_val

        if 'phase1' in self.mode:
            np.savez(self.conf.data_dir+f'{self.mode}.npz', X=self.X, Y=self.Y, mean_y=self.mean_data, std_y=self.std_data)
        elif 'phase2' in self.mode:
            np.savez(self.conf.data_dir+f'{self.mode}_lag{lag}.npz', X=self.X, Y=self.Y, mean_y=self.mean_data, std_y=self.std_data)
        
        # Convert to torch tensor
        self.X = torch.from_numpy(self.X).float().to(self.conf.device) # N-1, loci, nstates
        self.Y = torch.from_numpy(self.Y).float().to(self.conf.device) # N-1, loci, nstates
        
        
    def getLoader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=True)