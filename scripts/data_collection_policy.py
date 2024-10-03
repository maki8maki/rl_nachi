#!/usr/bin/env python3

import os
from datetime import datetime

import hydra
import numpy as np
import rospy
import torch as th
from config.config import SB3Config
from executer import SB3Executer
from omegaconf import OmegaConf
from utils import yes_no_input


@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "config"), config_name="sb3", version_base=None)
def main(_cfg: OmegaConf):
    rospy.init_node("rl_nachi")

    now = datetime.now().strftime("%Y%m%d-%H%M")

    cfg = SB3Config.convert(_cfg)
    print(f"\n{cfg}\n")

    executer = SB3Executer(cfg)
    del _cfg, cfg

    if not yes_no_input():
        exit()

    images = []
    poses = []

    while len(images) < 100:
        try:
            state = executer.get_state()
            ac, _ = executer.rl_model.predict(th.tensor(state), deterministic=True)
            executer.set_action(ac)
            images.append(executer.get_image())
            poses.append(executer.get_robot_state().copy())
        except AssertionError as e:
            print(e)
            executer.env.set_initial_position()

    dirs = os.path.join(os.path.dirname(__file__), "data", now)
    os.makedirs(dirs, exist_ok=True)

    images = np.array(images)
    poses = np.array(poses)

    np.save(os.path.join(dirs, "images"), images)
    np.save(os.path.join(dirs, "poses"), poses)

    executer.close()


if __name__ == "__main__":
    main()
