#!/usr/bin/env python3

import os

import hydra
import rospy
from config.config import SB3DAConfig
from executer import SB3DAExecuter
from omegaconf import OmegaConf
from utils import yes_no_input


@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "config"), config_name="sb3_da", version_base=None)
def main(_cfg: OmegaConf):
    rospy.init_node("rl_nachi")

    cfg = SB3DAConfig.convert(_cfg)
    print(f"\n{cfg}\n")

    if not yes_no_input():
        exit()

    executer = SB3DAExecuter(cfg)
    del _cfg, cfg

    if not yes_no_input():
        exit()

    executer.test(15)


if __name__ == "__main__":
    main()
