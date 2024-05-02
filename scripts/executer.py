from gymnasium import spaces
import numpy as np
import os
import torch as th
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

from .config.config import CombConfig
import rotations as rot
from utils import normalize

class Executer:
    cfg: CombConfig
    writer: SummaryWriter
    
    def __init__(self, cfg: CombConfig):
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=cfg.output_dir)
        
        self.make_aliases()
        
        self.fe_model.load_state_dict(th.load(os.path.join('./model', self.cfg.fe.model_name), map_location=self.cfg.device))
        self.rl_model.load_state_dict(th.load(os.path.join('./model', f'{self.cfg.basename}.pth'), map_location=self.cfg.device))
        self.fe_model.eval()
        self.rl_model.eval()
        
        # 画像取得の準備
        
        # その他
        self.robot_act_dim = 6
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.robot_act_dim,))
        self.observation_space = spaces.Box(
            low=np.array([-0.0, -1.0, -0.0, -np.pi, -np.pi, -np.pi]),
            high=np.array([2.0, 1.0, 2.0, np.pi, np.pi, np.pi]),
            dtype='float64'
        )
        
        self.rgb_imgs = []
        self.depth_imgs = []
        
    def make_aliases(self):
        self.fe_model = self.cfg.fe.model
        self.rl_model = self.cfg.rl.model
    
    def get_robot_state(self) -> np.ndarray:
        pass
    
    def get_image(self) -> np.ndarray:
        # self.rgb_imgs.append(rgb_img)
        # self.depth_imgs.append(depth_img)
        pass
    
    def get_state(self) -> Tuple[np.ndarray, th.Tensor]:
        img = self.get_image()
        normalized_img = normalize(img, 0, 255)
        rs = self.get_robot_state()
        normalized_rs = normalize(rs, self.observation_space.low, self.observation_space.high)
        tensor_image = th.tensor(self.cfg.fe.trans(normalized_img), dtype=th.float, device=self.cfg.device)
        hidden_state = self.fe_model.forward(tensor_image).cpu().squeeze().detach().numpy()
        state = np.concatenate([hidden_state, normalized_rs[:self.cfg.rl.obs_dim]])
        return rs, state
    
    def set_action(self, robot_state: np.ndarray, action: np.ndarray):
        # actionの整形など
        action = action.copy()
        if (self.cfg.rl.act_dim < self.robot_act_dim):
            action = np.concatenate([action, np.zeros((self.robot_act_dim-self.cfg.rl.act_dim,))])
        action = np.clip(action, -1, 1)
        assert action.shape == self.action_space.shape
        
        # unscale
        pos_ctrl, rot_ctrl = action[:3], action[3:]
        pos_ctrl *= 0.05 * 100 # mm
        rot_ctrl *= np.deg2rad(10) # rad
        
        # 目標の計算
        pos_cur, rot_cur = robot_state[:3], np.deg2rad(robot_state[3:])
        pos_target = pos_cur + pos_ctrl
        mat_target = rot.add_rot_mat(rot.euler2mat(rot_cur), rot.add_rot_mat(rot_ctrl))
        rot_target = np.rad2deg(rot.mat2euler(mat_target)) # deg
        target = np.concatenate(pos_target, rot_target)
        
        # 指令の送信
    
    def is_done(self) -> bool:
        pass
    
    def main_loop(self):
        done = False
        while not done:
            # Agent用の状態を取得
            rs, state = self.get_state()
            ac = self.rl_model.get_action(state, deterministic=True)
            self.set_action(rs, ac)
            done = self.is_done()
    
    def close(self):
        # (N, T, C, H, W)にself.rgb_imgsとself.depth_imgsを変換してadd_video
        self.writer.flush()
        self.writer.close()
    
    def __call__(self):
        self.main_loop()
        self.close()
