#!/usr/bin/env python3

import os

import hydra
from ament_index_python.packages import get_package_share_directory
from omegaconf import OmegaConf

from .config import CombConfig
from .executer import Executer
from .utils import yes_no_input


@hydra.main(
    config_path=os.path.join(get_package_share_directory("rl_nachi"), "rl_nachi", "config"),
    config_name="config",
    version_base=None,
)
def main(_cfg: OmegaConf):
    cfg = CombConfig.convert(_cfg)
    print(f"\n{cfg}\n")

    if not yes_no_input():
        exit()

    executer = Executer(cfg)
    del _cfg, cfg

    if not yes_no_input():
        exit()

    executer.test(10)


if __name__ == "__main__":
    main()
