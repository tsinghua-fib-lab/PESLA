from omegaconf import OmegaConf
import warnings; warnings.filterwarnings('ignore')

from model import PESLA_4_Well, PESLA_Homeodomain, PESLA_SSWM
from utils import *


def pipeline():

    # Read config
    conf = OmegaConf.load('config.yaml')
    
    # Set data and log directory
    if conf.system == '_4_Well':
        code_dim = conf.PESLA_4_Well.code_dim
        K = conf.PESLA_4_Well.K
        conf.log_dir = f'logs/{conf.system}/{conf.model}/seed{conf.seed}-ratio{conf.data_ratio}-cdim{code_dim}-K{K}/'
        conf.data_dir = f'data/{conf.system}/'
    elif conf.system == 'Homeodomain':
        code_dim = conf.PESLA_Homeodomain.code_dim
        K = conf.PESLA_Homeodomain.K
        conf.log_dir = f'logs/{conf.system}/{conf.model}/seed{conf.seed}-ratio{conf.data_ratio}-cdim{code_dim}-K{K}/'
        conf.data_dir = f'data/{conf.system}/'
    elif conf.system == 'SSWM':
        code_dim = conf.PESLA_SSWM.code_dim
        K = conf.PESLA_SSWM.K
        conf.log_dir = f'logs/{conf.system}_{conf.SSWM.states}/{conf.model}/seed{conf.seed}-ratio{conf.data_ratio}-cdim{code_dim}-K{K}/'
        conf.data_dir = f'data/{conf.system}_{conf.SSWM.states}/'
    
    # Set random seed and cpu number
    set_cpu_num(conf.cpu_num)
    seed_everything(conf.seed)
    
    # Model
    if conf.system == '_4_Well':
        model = PESLA_4_Well(conf)
    elif conf.system == 'Homeodomain':
        model = PESLA_Homeodomain(conf)
    elif conf.system == 'SSWM':
        model = PESLA_SSWM(conf)
    
    # Train
    model.fit()
    
    # Test
    model.test()
    

if __name__ == '__main__':
    pipeline()