#!/usr/bin/env python3

import os

import hydra
import rospy
from config.config import SB3Config
from executer import SB3Executer
from omegaconf import OmegaConf
from utils import yes_no_input


@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "config"), config_name="sb3", version_base=None)
def main(_cfg: OmegaConf):
    rospy.init_node("rl_nachi")

    cfg = SB3Config.convert(_cfg)
    print(f"\n{cfg}\n")

    if not yes_no_input():
        exit()

    executer = SB3Executer(cfg)
    del _cfg, cfg

    if not yes_no_input():
        executer.close()
        exit()

    executer.test(15)


if __name__ == "__main__":
    main()
