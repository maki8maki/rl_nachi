#!/usr/bin/env python3

import os
from datetime import datetime

import numpy as np
import rospy
from env import NachiEnv
from utils import yes_no_input


def main():
    rospy.init_node("data_collection")

    now = datetime.now().strftime("%Y%m%d-%H%M")

    env = NachiEnv()
    env.set_initial_position()

    if not yes_no_input():
        env.close()
        exit()

    images = []
    poses = []

    while len(images) < 3:
        try:
            env.set_action(np.random.uniform(-1.0, 1.0, (6,)))
            img = np.concatenate([env.rgb_image, np.expand_dims(env.depth_image, 2)], axis=2)
            images.append(img)
            env.update_robot_state()
            poses.append(env.tool_pose.copy())
        except AssertionError as e:
            print(e)
            env.set_initial_position()

    dirs = os.path.join(os.path.dirname(__file__), "data", now)
    os.makedirs(dirs, exist_ok=True)

    images = np.array(images)
    poses = np.array(poses)

    np.save(os.path.join(dirs, "images"), images)
    np.save(os.path.join(dirs, "poses"), poses)

    env.close()


if __name__ == "__main__":
    main()
