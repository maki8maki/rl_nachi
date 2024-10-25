import os

import numpy as np
import torch as th
from config.config import CombConfig, SB3Config, SB3DAConfig
from env import IMAGE_MAX, IMAGE_MIN, NachiEnv
from torch.utils.tensorboard import SummaryWriter
from utils import normalize


class Executer:
    cfg: CombConfig
    writer: SummaryWriter

    def __init__(self, cfg: CombConfig):
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=cfg.output_dir)

        self.make_aliases()

        model_dir = os.path.join(os.path.dirname(__file__), "model")

        self.fe_model.load_state_dict(
            th.load(os.path.join(model_dir, self.cfg.fe.model_name), map_location=self.cfg.device)
        )
        self.rl_model.load_state_dict(
            th.load(os.path.join(model_dir, f"{self.cfg.basename}.pth"), map_location=self.cfg.device)
        )
        self.fe_model.eval()
        self.rl_model.eval()

        self.env = NachiEnv()

        # その他
        self.steps = 0

    def make_aliases(self):
        self.fe_model = self.cfg.fe.model
        self.rl_model = self.cfg.model

    def get_robot_state(self) -> np.ndarray:
        self.env.update_robot_state()
        rs = self.env.tool_pose

        self.writer.add_tensor("position", th.tensor(rs, dtype=th.float), self.steps)
        return rs

    def get_image(self) -> np.ndarray:
        rgb = self.env.rgb_image
        depth = self.env.depth_image
        image = np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2)
        self.writer.add_image("rgb/original/raw", rgb, self.steps, dataformats="HWC")
        self.writer.add_image("depth/original/raw", depth, self.steps, dataformats="HW")
        return image

    def get_tensor_image(self, img: np.ndarray) -> th.Tensor:
        tensor_image = th.tensor(self.cfg.fe.trans(img), dtype=th.float, device=self.cfg.device)
        self.logging_image("rgb/original", tensor_image[:3])
        self.logging_image("depth/original", tensor_image[3:])
        return tensor_image

    def get_state(self) -> np.ndarray:
        img = self.get_image()
        normalized_img = normalize(img, IMAGE_MIN, IMAGE_MAX)
        rs = self.get_robot_state()
        normalized_rs = normalize(rs, self.env.observation_space.low, self.env.observation_space.high)
        tensor_image = self.get_tensor_image(normalized_img)
        hidden_state, recon_imgs = self.fe_model.forward(tensor_image.unsqueeze(0), return_pred=True)
        state = np.concatenate([hidden_state.cpu().squeeze().detach().numpy(), normalized_rs])

        # log
        recon_imgs = recon_imgs.squeeze()
        self.logging_image("rgb/reconstructed", recon_imgs[:3])
        self.logging_image("depth/reconstructed", recon_imgs[3:])
        return state

    def set_action(self, action: np.ndarray):
        # actionの整形など
        action = action.copy()
        action = np.clip(action, -1, 1)
        assert action.shape == self.env.action_space.shape

        self.writer.add_tensor("action", th.tensor(action, dtype=th.float), self.steps)

        self.env.set_action(action)

    def is_done(self) -> bool:
        pass

    def logging_image(self, tag: str, img: th.Tensor):
        self.writer.add_image(tag, img * 0.5 + 0.5, self.steps)

    def main_loop(self):
        done = False
        while not done:
            self.steps += 1
            state = self.get_state()  # Agent用の状態を取得
            ac = self.rl_model.get_action(state, deterministic=True)
            self.set_action(ac)
            done = self.is_done()

    def close(self):
        self.writer.flush()
        self.writer.close()
        self.env.close()

    def test_loop(self, loop_num: int):
        for _ in range(loop_num):
            print(self.steps)
            self.steps += 1
            state = self.get_state()
            ac = self.rl_model.get_action(state, deterministic=True)
            self.set_action(ac)

    def test(self, loop_num: int):
        self.env.set_initial_position()
        try:
            self.test_loop(loop_num)
        finally:
            self.close()

    def __call__(self):
        self.env.set_initial_position()
        try:
            self.main_loop()
        finally:
            self.close()


class SB3Executer(Executer):
    cfg: SB3Config

    def __init__(self, cfg: SB3Config):
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=cfg.output_dir)

        self.make_aliases()

        self.fe_model.eval()
        self.rl_model.set_training_mode(False)

        self.env = NachiEnv()

        # その他
        self.steps = 0

    def make_aliases(self):
        self.fe_model = self.cfg.fe.model
        self.rl_model = self.cfg.model

    def main_loop(self):
        done = False
        while not done:
            self.steps += 1
            state = self.get_state()  # Agent用の状態を取得
            ac, _ = self.rl_model.predict(th.tensor(state), deterministic=True)
            self.set_action(ac)
            done = self.is_done()

    def test_loop(self, loop_num: int):
        for _ in range(loop_num):
            print(self.steps)
            self.steps += 1
            state = self.get_state()
            ac, _ = self.rl_model.predict(th.tensor(state), deterministic=True)
            self.set_action(ac)


class SB3DAExecuter(SB3Executer):
    cfg: SB3DAConfig

    def __init__(self, cfg: SB3DAConfig):
        super().__init__(cfg)

        self.da_model.eval()

    def make_aliases(self):
        super().make_aliases()
        self.da_model = self.cfg.da.model

    def get_tensor_image(self, img: np.ndarray) -> th.Tensor:
        tensor_image = super().get_tensor_image(img)
        fake_sim_img = self.da_model.netG_B(tensor_image.unsqueeze(0))
        fake_sim_img = fake_sim_img.squeeze()
        self.logging_image("rgb/fake_sim", fake_sim_img[:3])
        self.logging_image("depth/fake_sim", fake_sim_img[3:])
        return fake_sim_img
