import os
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import scipy
from sklearn.linear_model import RANSACRegressor
import scienceplots
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


plt.style.use(['ieee', 'science', 'no-latex'])
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams["font.family"] = 'Arial'


def set_cpu_num(cpu_num: int = 1):
    if cpu_num <= 0: return
    
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def seed_everything(seed: int = 42):
    # Set the random seed for Python's built-in random module
    random.seed(seed)
    
    # Set the random seed for NumPy
    np.random.seed(seed)
    
    # Set the random seed for torch operations
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    print(model)
    print("Number of parameters: {:,}".format(count_parameters(model)))

    print("\nParameter details:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name, parameter.shape, parameter.device, parameter.dtype, parameter.numel())


def calculate_jsd_ndarray(X, Y):
    B, N = X.shape
    jsd_array = np.zeros(B)
    
    for i in range(B):
        x_prob = X[i] / (np.sum(X[i]) + 1e-12)
        y_prob = Y[i] / (np.sum(Y[i]) + 1e-12)
        
        if np.sum(X[i]) == 0. and np.sum(Y[i]) == 0.:
            jsd_array[i] = 0.
        else:
            if np.sum(X[i]) == 0.:
                x_prob = np.ones_like(x_prob) / N
            if np.sum(Y[i]) == 0.:
                y_prob = np.ones_like(y_prob) / N
            
            jsd_array[i] = jensenshannon(x_prob, y_prob, base=2) ** 2

    return jsd_array


def test_4_well_energy(model, true_energy, sample_energy, lag):
    
    # Correlation
    spearmans_rho = scipy.stats.spearmanr(true_energy, sample_energy)
    print(f'Lag={lag} | Spearman\'s ρ={spearmans_rho.correlation:.4f}')
    
    # RANSAC
    ransac = RANSACRegressor()
    ransac.fit(true_energy.reshape(-1, 1), sample_energy)
    slope_ransac = ransac.estimator_.coef_[0]
    intercept_ransac = ransac.estimator_.intercept_
    print(f'Lag={lag} | RANSAC: y = {slope_ransac:.4f}x + {intercept_ransac:.4f}')


def test_4_well(model, traj_num, n_samples, grid_num, lag):
    # initial state
    init_state = np.array([[0, 0]])
    
    # GT
    from deeptime.data import quadruple_well_asymmetric
    system = quadruple_well_asymmetric(n_steps=100, h=1e-4)
    gt_trajs = []
    for seed in tqdm(range(10, 10+traj_num)):
        gt_traj = system.trajectory(init_state, n_samples, seed=seed)
        gt_trajs.append(gt_traj)
    gt_trajs = np.array(gt_trajs)[:, -n_samples:] # (traj_num, n_samples, 2)
    
    # Energy
    gt_energy = system.potential(gt_trajs.reshape(-1, 2))
    pred_energy = model.Energy(torch.tensor(gt_trajs.reshape(-1, 2), dtype=torch.float32).to(model.conf.device)).squeeze().cpu().detach().numpy()
    test_4_well_energy(model, gt_energy, pred_energy, lag)
        
    # Generate sample
    with torch.no_grad():
        init_state = torch.tensor(init_state, dtype=torch.float32).to(model.conf.device).repeat(traj_num, 1)
        sample_trajs, sample_traj_code_idx, sample_traj_P = model.sample(init_state, n_samples//lag)
        sample_trajs = sample_trajs.cpu().detach().numpy()[:, -n_samples//lag:] # (traj_num, n_samples//lag, 2)
        sample_traj_code_idx = sample_traj_code_idx.cpu().detach().numpy()[:, -n_samples//lag:] # (traj_num, n_samples//lag)
        sample_traj_P = sample_traj_P.cpu().detach().numpy()[:, -n_samples//lag:] # (traj_num, n_samples//lag, N)
    gt_trajs = gt_trajs[:, -n_samples::lag] # (traj_num, n_samples//lag, 2)
    
    # Meshgrid discretization
    grid_size = 3.6 / (grid_num-1)
    gt_state_freq, sample_state_freq = np.zeros((traj_num, grid_num, grid_num)), np.zeros((traj_num, grid_num, grid_num))
    gt_trans_freq, sample_trans_freq = np.zeros((traj_num, grid_num*grid_num, grid_num*grid_num)), np.zeros((traj_num, grid_num*grid_num, grid_num*grid_num))
    gt_trajs_grid, sample_trajs_grid = torch.zeros(gt_trajs.shape[0], gt_trajs.shape[1], dtype=int), torch.zeros(sample_trajs.shape[0], sample_trajs.shape[1], dtype=int)
    
    invaild_num = 0
    for i, traj in enumerate(gt_trajs):
        for j, (x, y) in enumerate(traj):
            if x > -1.8 and x < 1.8 and y > -1.8 and y < 1.8:
                x_idx = np.floor((x-(-1.8)) / grid_size).astype(int) + 1
                y_idx = np.floor((y-(-1.8)) / grid_size).astype(int) + 1
            else:
                invaild_num += 1
                continue
            
            state_idx = x_idx + y_idx * grid_num
            gt_state_freq[i, x_idx, y_idx] += 1
            gt_trajs_grid[i, j] = state_idx
            
            if j == 0:
                last_state_idx = x_idx + y_idx * grid_num
            else:
                gt_trans_freq[i, last_state_idx, state_idx] += 1
                last_state_idx = x_idx + y_idx * grid_num
    print(f'Invalid number: {invaild_num}')
    
    invaild_num = 0
    for i, traj in enumerate(sample_trajs):
        for j, (x, y) in enumerate(traj):
            if x > -1.8 and x < 1.8 and y > -1.8 and y < 1.8:
                x_idx = np.floor((x-(-1.8)) / grid_size).astype(int) + 1
                y_idx = np.floor((y-(-1.8)) / grid_size).astype(int) + 1
            else:
                invaild_num += 1
                continue
            
            state_idx = x_idx + y_idx * grid_num
            sample_state_freq[i, x_idx, y_idx] += 1
            sample_trajs_grid[i, j] = state_idx
            
            if j == 0:
                last_state_idx = x_idx + y_idx * grid_num
            else:
                sample_trans_freq[i, last_state_idx, state_idx] += 1
                last_state_idx = x_idx + y_idx * grid_num
    print(f'Invalid number: {invaild_num}')
    
    # JSD of Marginal State Probability
    MJS_list = calculate_jsd_ndarray(gt_state_freq.reshape(traj_num,-1), sample_state_freq.reshape(traj_num,-1))
    print(f'MSJ@{lag}: {MJS_list.mean()}')
    
    # JSD of Transition Probability
    TJS_list = calculate_jsd_ndarray(gt_trans_freq.reshape(traj_num,-1), sample_trans_freq.reshape(traj_num,-1))
    print(f'TJS@{lag}: {TJS_list.mean()}')
    




class EnergyLandscape:
    def __init__(self, energy_reshaped, bin_x, bin_y):
        self.energy_reshaped = energy_reshaped
        self.bin_x = bin_x
        self.bin_y = bin_y
    
    def query_energy(self, x, y):
        x = np.digitize(x, bins=self.bin_x)
        y = np.digitize(y, bins=self.bin_y)
        return self.energy_reshaped[y-1, x-1]



def test_protein(model, grid_num, traj_num, gt_traj_tica, sample_traj, lag):
    # Meshgrid discretization
    xmin, xmax, ymin, ymax = gt_traj_tica[:, 0].min(), gt_traj_tica[:, 0].max(), gt_traj_tica[:, 1].min(), gt_traj_tica[:, 1].max()
    grid_size_x, grid_size_y = (xmax-xmin)/grid_num, (ymax-ymin)/grid_num

    gt_traj_tica, sample_traj = gt_traj_tica[np.newaxis, :], sample_traj[np.newaxis, :]
    gt_state_freq, sample_state_freq = np.zeros((traj_num, grid_num, grid_num)), np.zeros((traj_num, grid_num, grid_num))
    gt_trans_freq, sample_trans_freq = np.zeros((traj_num, grid_num*grid_num, grid_num*grid_num)), np.zeros((traj_num, grid_num*grid_num, grid_num*grid_num))
    gt_trajs_grid, sample_trajs_grid = torch.zeros(gt_traj_tica.shape[0], gt_traj_tica.shape[1], dtype=int), torch.zeros(sample_traj.shape[0], sample_traj.shape[1], dtype=int)

    invaild_num = 0
    for i, traj in enumerate(gt_traj_tica):
        for j, (x, y) in enumerate(traj):
            if x > xmin and x < xmax and y > ymin and y < ymax:
                x_idx = np.floor((x-(xmin)) / grid_size_x).astype(int)
                y_idx = np.floor((y-(ymin)) / grid_size_y).astype(int)
            else:
                invaild_num += 1
                continue
            
            state_idx = x_idx + y_idx * grid_num
            gt_state_freq[i, x_idx, y_idx] += 1
            gt_trajs_grid[i, j] = state_idx
            
            if j == 0:
                last_state_idx = x_idx + y_idx * grid_num
            else:
                gt_trans_freq[i, last_state_idx, state_idx] += 1
                last_state_idx = x_idx + y_idx * grid_num
    print(f'Invalid number: {invaild_num}')

    invaild_num = 0
    for i, traj in enumerate(sample_traj):
        for j, (x, y) in enumerate(traj):
            if x > xmin and x < xmax and y > ymin and y < ymax:
                x_idx = np.floor((x-(xmin)) / grid_size_x).astype(int)
                y_idx = np.floor((y-(ymin)) / grid_size_y).astype(int)
            else:
                invaild_num += 1
                continue
            
            state_idx = x_idx + y_idx * grid_num
            sample_state_freq[i, x_idx, y_idx] += 1
            sample_trajs_grid[i, j] = state_idx
            
            if j == 0:
                last_state_idx = x_idx + y_idx * grid_num
            else:
                sample_trans_freq[i, last_state_idx, state_idx] += 1
                last_state_idx = x_idx + y_idx * grid_num
    print(f'Invalid number: {invaild_num}')

    # JSD of Marginal State Probability
    MJS_list = calculate_jsd_ndarray(gt_state_freq.reshape(traj_num,-1), sample_state_freq.reshape(traj_num,-1))
    print(f'MSJ@{lag}: {MJS_list.mean()}')

    # JSD of Transition Probability
    TJS_list = calculate_jsd_ndarray(gt_trans_freq.reshape(traj_num,-1), sample_trans_freq.reshape(traj_num,-1))
    print(f'TJS@{lag}: {TJS_list.mean()}')
    
    
    

def test_sswm_energy(model, sample_coords, gt_trajs, true_energy, sample_energy, code_idxs):
    # Correlation
    spearmans_rho = scipy.stats.spearmanr(true_energy, sample_energy)
    print(f'Spearman\'s ρ={spearmans_rho.correlation:.4f}')
    
    # RANSAC
    ransac = RANSACRegressor()
    ransac.fit(true_energy.reshape(-1, 1), sample_energy)
    slope_ransac = ransac.estimator_.coef_[0]
    intercept_ransac = ransac.estimator_.intercept_
    print(f'RANSAC: y = {slope_ransac:.4f}x + {intercept_ransac:.4f}')
    ransac_func = lambda x: slope_ransac * x + intercept_ransac
    
    return ransac_func



def test_sswm(model, traj_num, steps, lag=1):
    # GT System
    with open(model.conf.data_dir+'system.pkl', 'rb') as f:
        system = pickle.load(f)
    
    # initial state
    initial_genotypes = [list(np.random.choice(range(system.allele_count[0]), size=system.loci)) for _ in range(traj_num)]
        
    # GT
    gt_trajs = []
    for i, seed in tqdm(enumerate(range(1000, 1000+traj_num))):
        gt_traj = system.run(initial_genotype=initial_genotypes[i], steps=steps, seed=seed)
        gt_trajs.append(gt_traj)
    gt_trajs = np.array(gt_trajs) # (traj_num, steps, 2)

    # Sample
    xy = np.arange(0, model.conf.SSWM.states)
    sample_coords = np.dstack(np.meshgrid(xy, xy)).reshape(-1, 2)
    # one-hot
    gt_trajs_onehot = np.eye(model.conf.SSWM.states)[gt_trajs.reshape(-1, 2)] # (traj_num*steps, loci, N_states)
    sample_energy = model.Energy(torch.tensor(gt_trajs_onehot, dtype=torch.float32).to(model.conf.device)).squeeze().cpu().detach().numpy()
    sample_coords_onehot = np.eye(model.conf.SSWM.states)[sample_coords] # (traj_num*steps, loci, N_states)
    code_idx = model.encode(torch.tensor(sample_coords_onehot, dtype=torch.float32).to(model.conf.device)).cpu().detach().numpy()
    true_energy = np.zeros_like(sample_energy)
    for i in range(len(gt_trajs.reshape(-1, 2))):
        true_energy[i] = system.query_fitness(gt_trajs.reshape(-1, 2)[i])
    
    ransac_func = test_sswm_energy(model, sample_coords.reshape(-1, 2), gt_trajs.reshape(-1, 2), true_energy, sample_energy, code_idx)
    
    # Vis Landscape
    true_energy = np.zeros(len(sample_coords))
    ransac_energy = np.zeros((model.conf.SSWM.states, model.conf.SSWM.states))
    pred_energy = np.zeros((model.conf.SSWM.states, model.conf.SSWM.states))
    for i in range(len(sample_coords)):
        true_energy[i] = system.query_fitness(sample_coords[i])
        ransac_energy[sample_coords[i, 0], sample_coords[i, 1]] = ransac_func(true_energy[i])
        pred_energy[sample_coords[i, 0], sample_coords[i, 1]] = model.Energy(torch.tensor(np.eye(model.conf.SSWM.states)[sample_coords[i].reshape(-1, 2)], dtype=torch.float32).to(model.conf.device)).squeeze().cpu().detach().numpy()

    # Sample
    with torch.no_grad():
        init_state = torch.tensor(initial_genotypes) # (traj_num, loci)
        init_state_onehot = torch.eye(model.conf.SSWM.states)[init_state].to(model.conf.device) # (traj_num, loci, N_states)
        sample_trajs = []
        for batch_idx in range(traj_num//100):
            batch_init_state_onehot = init_state_onehot[batch_idx*100:(batch_idx+1)*100]
            sample_trajs.append(model.sample(batch_init_state_onehot, steps//lag)[:, -steps//lag:])
    sample_trajs = np.concatenate(sample_trajs) # (traj_num, steps//lag, loci)
    gt_trajs = gt_trajs[:, -steps::lag] # (traj_num, steps//lag, loci)
    states = model.conf.SSWM.states
    
    # Meshgrid discretization
    downsample = 8
    coarsen_states = states // downsample
    gt_state_freq, sample_state_freq = np.zeros((traj_num, coarsen_states, coarsen_states)), np.zeros((traj_num, coarsen_states, coarsen_states))
    gt_trans_freq, sample_trans_freq = np.zeros((traj_num, coarsen_states*coarsen_states, coarsen_states*coarsen_states)), np.zeros((traj_num, coarsen_states*coarsen_states, coarsen_states*coarsen_states))
    gt_trajs_grid, sample_trajs_grid = torch.zeros(gt_trajs.shape[0], gt_trajs.shape[1], dtype=int), torch.zeros(sample_trajs.shape[0], sample_trajs.shape[1], dtype=int)
    
    invaild_num = 0
    for i, traj in enumerate(gt_trajs):
        for j, (x, y) in enumerate(traj):
            if x >= 0 and x < states and y >= 0 and y < states:
                x_idx = int(x) // downsample
                y_idx = int(y) // downsample
            else:
                invaild_num += 1
                continue
            
            state_idx = x_idx + y_idx * coarsen_states
            gt_state_freq[i, x_idx, y_idx] += 1
            gt_trajs_grid[i, j] = state_idx
            
            if j == 0:
                last_state_idx = x_idx + y_idx * coarsen_states
            else:
                gt_trans_freq[i, last_state_idx, state_idx] += 1
                last_state_idx = x_idx + y_idx * coarsen_states
    print(f'Invalid number: {invaild_num}')
    
    invaild_num = 0
    for i, traj in enumerate(sample_trajs):
        for j, (x, y) in enumerate(traj):
            if x >= 0 and x < states and y >= 0 and y < states:
                x_idx = int(x) // downsample
                y_idx = int(y) // downsample
            else:
                invaild_num += 1
                continue
            
            state_idx = x_idx + y_idx * coarsen_states
            sample_state_freq[i, x_idx, y_idx] += 1
            sample_trajs_grid[i, j] = state_idx
            
            if j == 0:
                last_state_idx = x_idx + y_idx * coarsen_states
            else:
                sample_trans_freq[i, last_state_idx, state_idx] += 1
                last_state_idx = x_idx + y_idx * coarsen_states
    print(f'Invalid number: {invaild_num}')
    
    # JSD of Marginal State Probability
    MJS_list = calculate_jsd_ndarray(gt_state_freq.reshape(traj_num,-1), sample_state_freq.reshape(traj_num,-1))
    print(f'JSD of Marginal State Probability: {MJS_list.mean()}')
    
    # JSD of Transition Probability
    TJS_list = calculate_jsd_ndarray(gt_trans_freq.reshape(traj_num,-1), sample_trans_freq.reshape(traj_num,-1))
    print(f'JSD of Transition Probability: {TJS_list.mean()}')
    
    
    
    
def test_4_well_energy_pretrain(model, traj_num, n_samples):
    os.makedirs(model.conf.log_dir+'test/', exist_ok=True)
    
    # initial state
    init_state = np.array([[0, 0]])
        
    # GT
    from deeptime.data import quadruple_well_asymmetric
    system = quadruple_well_asymmetric(n_steps=100, h=1e-4)
    gt_trajs = []
    for seed in tqdm(range(10, 10+traj_num)):
        gt_traj = system.trajectory(init_state, n_samples, seed=seed)
        gt_trajs.append(gt_traj)
    gt_trajs = np.array(gt_trajs)[:, -n_samples:] # (traj_num, n_samples, 2)
    
    # Energy
    true_energy = system.potential(gt_trajs.reshape(-1, 2))
    sample_energy = model.Energy(torch.tensor(gt_trajs.reshape(-1, 2), dtype=torch.float32).to(model.conf.device)).squeeze().cpu().detach().numpy()
    sample_coords = gt_trajs.reshape(-1, 2)
    
    # Correlation
    spearmans_rho = scipy.stats.spearmanr(true_energy, sample_energy)
    print(f'Spearman\'s ρ={spearmans_rho.correlation:.4f}')
    
    # RANSAC
    ransac = RANSACRegressor()
    ransac.fit(true_energy.reshape(-1, 1), sample_energy)
    slope_ransac = ransac.estimator_.coef_[0]
    intercept_ransac = ransac.estimator_.intercept_
    residual_threshold = 1.
    print(f'Pretrain RANSAC: y = {slope_ransac:.4f}x + {intercept_ransac:.4f}')