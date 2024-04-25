import hydra
from omegaconf import OmegaConf

from config.config import CombConfig
from executer import Executer
from utils import yes_no_input

@hydra.main(config_path='config/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = CombConfig.convert(_cfg)
    print(f'\n{cfg}\n')
    
    executer = Executer(cfg)
    del _cfg, cfg
    
    if not yes_no_input():
        exit()
    
    executer()

if __name__ == '__main__':
    main()
