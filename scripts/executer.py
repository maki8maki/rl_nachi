import os
from typing import Tuple

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from .config.config import CombConfig
from .env import IMAGE_HEIGHT, IMAGE_MAX, IMAGE_MIN, IMAGE_WIDTH, NachiEnv
from .utils import normalize


class Executer:
    cfg: CombConfig
    writer: SummaryWriter

    def __init__(self, cfg: CombConfig):
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=cfg.output_dir)

        self.make_aliases()

        self.fe_model.load_state_dict(
            th.load(os.path.join("./model", self.cfg.fe.model_name), map_location=self.cfg.device)
        )
        self.rl_model.load_state_dict(
            th.load(os.path.join("./model", f"{self.cfg.basename}.pth"), map_location=self.cfg.device)
        )
        self.fe_model.eval()
        self.rl_model.eval()

        self.env = NachiEnv()

        # その他
        self.rgb_imgs = []
        self.depth_imgs = []
        self.steps = 0

    def make_aliases(self):
        self.fe_model = self.cfg.fe.model
        self.rl_model = self.cfg.rl.model

    def get_robot_state(self) -> np.ndarray:
        self.env.update_robot_state()
        rs = self.env.robot_state

        self.writer.add_tensor("position", th.tensor(rs, dtype=th.float), self.steps)
        return rs

    def get_image(self) -> np.ndarray:
        image = np.concatenate([self.env.rgb_image, self.env.depth_image], axis=2)
        self.rgb_imgs.append(self.env.rgb_image)
        self.depth_imgs.append(self.env.depth_image)
        return image

    def get_state(self) -> Tuple[np.ndarray, th.Tensor]:
        img = self.get_image()
        normalized_img = normalize(img, IMAGE_MIN, IMAGE_MAX)
        rs = self.get_robot_state()
        normalized_rs = normalize(rs, self.env.observation_space.low, self.env.observation_space.high)
        tensor_image = th.tensor(self.cfg.fe.trans(normalized_img), dtype=th.float, device=self.cfg.device)
        hidden_state = self.fe_model.forward(tensor_image).cpu().squeeze().detach().numpy()
        state = np.concatenate([hidden_state, normalized_rs[: self.cfg.rl.obs_dim]])
        return state

    def set_action(self, action: np.ndarray):
        # actionの整形など
        action = action.copy()
        if self.cfg.rl.act_dim < self.env.robot_act_dim:
            action = np.concatenate([action, np.zeros((self.env.robot_act_dim - self.cfg.rl.act_dim,))])
        action = np.clip(action, -1, 1)
        assert action.shape == self.env.action_space.shape

        self.writer.add_tensor("action", th.tensor(action, dtype=th.float), self.steps)

        self.env.set_action(action)

    def is_done(self) -> bool:
        pass

    def main_loop(self):
        done = False
        while not done:
            state = self.get_state()  # Agent用の状態を取得
            ac = self.rl_model.get_action(state, deterministic=True)
            self.set_action(ac)
            done = self.is_done()
            self.steps += 1

    def close(self):
        # 各ステップでの画像を保存
        rgb_tensor = th.tensor(np.array(self.rgb_imgs), dtype=th.uint8)
        rgb_tensor = th.reshape(rgb_tensor, (1, -1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.writer.add_video("rgb_images", rgb_tensor)
        depth_tensor = th.tensor(np.array(self.depth_imgs), dtype=th.uint8)
        depth_tensor = th.reshape(depth_tensor, (1, -1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.writer.add_video("depth_images", depth_tensor)

        self.writer.flush()
        self.writer.close()
        self.env.close()

    def __call__(self):
        self.main_loop()
        self.close()
