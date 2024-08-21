#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rospy
from env import NachiEnv
from google.protobuf.json_format import MessageToDict
from tensorboard.backend.event_processing.event_accumulator import (
    DEFAULT_SIZE_GUIDANCE,
    TENSORS,
    EventAccumulator,
    TensorEvent,
)
from utils import yes_no_input


def playback(log_dir, length=0):
    rospy.init_node("rl_nachi")

    log_files = [path for path in Path(log_dir).glob("events*") if path.is_file()]
    log_file = log_files[0]

    size_guidance = DEFAULT_SIZE_GUIDANCE
    size_guidance.update([(TENSORS, length)])

    event = EventAccumulator(str(log_file), size_guidance=size_guidance)
    event.Reload()

    tes: list[TensorEvent] = event.Tensors("action")
    # tes: list[TensorEvent] = event.Tensors("position")

    env = NachiEnv()
    env.set_initial_position()

    if not yes_no_input():
        exit()

    id = 0

    try:
        for te in tes:
            tensor_proto = te.tensor_proto
            action = MessageToDict(tensor_proto)["floatVal"]
            print(action)
            env.set_action(action)
            # position = np.array(MessageToDict(tensor_proto)["floatVal"])
            # position[3], position[5] = position[5], position[3]
            # target = np.concatenate([position[:3] * 1000, np.rad2deg(position[3:])])
            # print(target)
            # env.set_position_action(target)
            id += 1
            if id >= 5:
                break

        rospy.sleep(2)
    except rospy.exceptions.ROSInterruptException:
        pass
    env.close()


if __name__ == "__main__":
    log_dir = "src/rl_nachi/logs/SB3_SAC_trainedVAE/20240705-2028"
    playback(log_dir)
