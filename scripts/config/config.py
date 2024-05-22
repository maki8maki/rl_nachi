import dataclasses
from copy import deepcopy

import dacite
import hydra
import torch
import torch.nn as nn
from absl import logging
from agents import utils
from agents.DCAE import DCAE
from agents.SAC import SAC
from omegaconf import OmegaConf


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

    def __post_init__(self, _model, _trans):
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
class RLConfig:
    obs_dim: int = 6
    act_dim: int = 6
    _model: dataclasses.InitVar[dict] = None
    model: utils.RL = dataclasses.field(default=None)

    def __post_init__(self, _model):
        if _model is None:
            self.model = SAC(obs_dim=self.obs_dim, act_dim=self.act_dim)

    def convert(self, _cfg: OmegaConf):
        self_copy = deepcopy(self)
        if _cfg._model:
            self_copy.model = hydra.utils.instantiate(_cfg._model)
        return self_copy


@dataclasses.dataclass
class CombConfig:
    fe: FEConfig
    rl: RLConfig
    basename: str
    position_random: bool = False
    posture_random: bool = False
    save_anim_num: int = dataclasses.field(default=10, repr=False)
    device: str = "cpu"
    fe_with_init: dataclasses.InitVar[bool] = True
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, fe_with_init):
        if self.device == "cpu":
            logging.warning("You are using CPU!!")
        if self.device == "cuda" and not torch.cuda.is_available():
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
        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        if _cfg.rl._model:
            _cfg.rl._model.obs_dim = cfg.rl.obs_dim + cfg.fe.hidden_dim
        cfg.rl = cfg.rl.convert(OmegaConf.create(_cfg.rl))
        cfg.fe.model.to(cfg.device)
        cfg.rl.model.to(cfg.device)
        cfg.basename = _cfg.basename + ("_r" if cfg.position_random else "_s") + ("r" if cfg.posture_random else "s")
        return cfg
