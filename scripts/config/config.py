import dataclasses
import importlib
import os
from copy import deepcopy
from typing import Optional, Type

import dacite
import hydra
import torch as th
import torch.nn as nn
from absl import logging
from agents import utils
from agents.DCAE import DCAE
from gymnasium.spaces import Box
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


@dataclasses.dataclass
class FEConfig:
    img_width: int = 108
    img_height: int = 72
    img_channel: int = 4
    hidden_dim: int = 20
    model_name: str = ""
    _model: dataclasses.InitVar[dict] = None
    model: utils.FE = dataclasses.field(default=None)
    _trans: dataclasses.InitVar[dict] = None
    trans: nn.Module = dataclasses.field(default=None, repr=False)

    def __post_init__(self, _model: Optional[dict], _trans: Optional[dict]):
        if _model is None:
            self.model = DCAE(
                img_height=self.img_height,
                img_width=self.img_width,
                img_channel=self.img_channel,
                hidden_dim=self.hidden_dim,
            )
        if _trans is None:
            self.trans = utils.MyTrans(img_width=self.img_width, img_height=self.img_height)

    def convert(self, _cfg: OmegaConf):
        self_copy = deepcopy(self)
        if _cfg._model:
            self_copy.model = hydra.utils.instantiate(_cfg._model)
        if _cfg._trans:
            self_copy.trans = hydra.utils.instantiate(_cfg._trans)
        return self_copy


@dataclasses.dataclass
class CombConfig:
    fe: FEConfig
    basename: str
    _model: dataclasses.InitVar[dict] = None
    model: utils.RL = dataclasses.field(default=None)
    position_random: bool = False
    posture_random: bool = False
    device: str = "cpu"
    fe_with_init: dataclasses.InitVar[bool] = True
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, fe_with_init: bool):
        if self.device == "cpu":
            logging.warning("You are using CPU!!")
        if self.device == "cuda" and not th.cuda.is_available():
            self.device = "cpu"
            logging.warning("Device changed to CPU!!")
        if self.fe.model is not None:
            self.fe.model.to(self.device)
        if self.rl.model is not None:
            self.rl.model.to(self.device)
        if fe_with_init:
            init = "w-init"
        else:
            init = "wo-init"
        if self.position_random:
            position_random = "r"
        else:
            position_random = "s"
        if self.posture_random:
            posture_random = "r"
        else:
            posture_random = "s"
        self.fe.model_name = self.fe.model_name.replace(".pth", f"_{position_random}{posture_random}_{init}.pth")
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        cfg.basename = _cfg.basename + ("_r" if cfg.position_random else "_s") + ("r" if cfg.posture_random else "s")

        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        cfg.fe.model.load_state_dict(th.load(os.path.join(MODEL_DIR, cfg.fe.model_name)))
        cfg.fe.model.to(cfg.device)

        obs_dim = cfg.env.observation_space.low.size + cfg.fe.hidden_dim
        act_dim = cfg.env.action_space.low.size
        cfg.model = hydra.utils.instantiate(_cfg._model, obs_dim=obs_dim, act_dim=act_dim)
        cfg.model.load_state_dict(th.load(os.path.join(MODEL_DIR, f"{cfg.basename}.pth")))
        cfg.model.to(cfg.device)

        th.save(cfg.fe.model.state_dict(), os.path.join(cfg.output_dir, cfg.fe.model_name))
        cfg.model.save(os.path.join(cfg.output_dir, f"{cfg.basename}.pth"))

        return cfg


@dataclasses.dataclass
class SB3Config:
    fe: FEConfig
    basename: str
    position_random: bool = False
    posture_random: bool = False
    fe_with_init: dataclasses.InitVar[bool] = True
    device: str = "cpu"
    model_class: dataclasses.InitVar[str] = "stable_baselines3.SAC"
    model: BasePolicy = dataclasses.field(default=None, repr=False)
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, fe_with_init: bool, model_class: str):
        if self.device == "cpu":
            logging.warning("You are using CPU!!")
        if self.device == "cuda" and not th.cuda.is_available():
            self.device = "cpu"
            logging.warning("Device changed to CPU!!")
        if fe_with_init:
            init = "w-init"
        else:
            init = "wo-init"
        if self.position_random:
            position_random = "r"
        else:
            position_random = "s"
        if self.posture_random:
            posture_random = "r"
        else:
            posture_random = "s"
        self.basename += f"_{position_random}{posture_random}"
        module_name, class_name = model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls: Type[BaseAlgorithm] = getattr(module, class_name)
        algo: BaseAlgorithm = cls.load(
            path=os.path.join(MODEL_DIR, self.basename),
            custom_objects={"action_space": Box(-1.0, 1.0, (6,))},
        )
        self.model = algo.policy
        self.fe.model_name = self.fe.model_name.replace(".pth", f"_{position_random}{posture_random}_{init}.pth")
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        cfg.fe.model.load_state_dict(th.load(os.path.join(MODEL_DIR, cfg.fe.model_name)))
        cfg.fe.model.to(device=cfg.device)
        cfg.model.to(device=cfg.device)

        th.save(cfg.fe.model.state_dict(), os.path.join(cfg.output_dir, cfg.fe.model_name))
        cfg.model.save(os.path.join(cfg.output_dir, f"{cfg.basename}.zip"))
        return cfg
