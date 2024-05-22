#!/usr/bin/env python3

import os

import hydra
import rospy
from config.config import CombConfig
from executer import Executer
from omegaconf import OmegaConf
from utils import yes_no_input


@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "config"), config_name="config", version_base=None)
def main(_cfg: OmegaConf):
    rospy.init_node("rl_nachi")

    cfg = CombConfig.convert(_cfg)
    print(f"\n{cfg}\n")

    executer = Executer(cfg)
    del _cfg, cfg

    if not yes_no_input():
        exit()

    executer.test(10)


if __name__ == "__main__":
    main()
