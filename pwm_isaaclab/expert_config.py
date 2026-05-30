from __future__ import annotations

import os
import tempfile

import yaml
from yacs.config import CfgNode as CN

try:
    from pwm_isaaclab.utils import load_config
except ImportError:
    from utils import load_config


EXPERT_SECTION_KEYS = {"expert", "replay", "logging"}


def _merge_dict_into_cfg(cfg_node, values):
    if not values:
        return
    cfg_node.merge_from_other_cfg(CN(values, new_allowed=True))


def _raw_yaml(config_path):
    with open(os.path.abspath(os.path.expanduser(config_path)), "r", encoding="utf-8") as fin:
        return yaml.safe_load(fin) or {}


def _load_legacy_without_expert_sections(config_path, raw):
    legacy_raw = {key: value for key, value in raw.items() if key not in EXPERT_SECTION_KEYS}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=True, encoding="utf-8") as fout:
        yaml.safe_dump(legacy_raw, fout, sort_keys=False)
        fout.flush()
        return load_config(fout.name)


def load_expert_config(config_path):
    raw = _raw_yaml(config_path)
    conf = _load_legacy_without_expert_sections(config_path, raw)
    conf.defrost()

    conf.expert = CN(new_allowed=True)
    conf.expert.enabled = True
    conf.expert.path = None
    conf.expert.format = "npz"
    conf.expert.preload_to_replay = True
    conf.expert.pretrain_world_model = True
    conf.expert.pretrain_steps = 50000
    conf.expert.bc_init = True
    conf.expert.bc_steps = 20000
    conf.expert.bc_batch_size = None
    conf.expert.bc_lr = None
    conf.expert.online_bc_reg = False
    conf.expert.online_bc_weight = 0.0
    conf.expert.online_bc_decay = 1.0
    conf.expert.replace_random_prefill = True
    conf.expert.keep_expert_in_replay = True
    conf.expert.expert_ratio_online = 0.2
    conf.expert.validate_expert_shapes = True
    conf.expert.save_init_checkpoints = True
    conf.expert.skip_pretrain_if_checkpoint_exists = False
    conf.expert.load_pretrained_world_model_path = None
    conf.expert.load_bc_actor_path = None
    conf.expert.validation_ratio = 0.05
    conf.expert.log_interval = 100
    conf.expert.action_tolerance = 1e-4
    conf.expert.cost_head_mode = "hurdle"
    conf.expert.cost_target_source = "force_margin"
    conf.expert.cost_pipe_force_limit = 1.0
    conf.expert.cost_bottom_force_limit = 1.0
    conf.expert.cost_pipe_force_channels = [1, 4]
    conf.expert.cost_bottom_force_channels = [2, 5]
    conf.expert.cost_balanced_sequence_ratio = 0.3
    conf.expert.cost_loss_type = "auto"
    conf.expert.cost_loss_weight = 2.0
    conf.expert.cost_cls_weight = 1.0
    conf.expert.cost_reg_weight = 1.0
    conf.expert.cost_prior_loss_weight = 0.5
    conf.expert.cost_focal_alpha = 0.75
    conf.expert.cost_focal_gamma = 2.0
    conf.expert.cost_pos_weight_max = 200.0
    conf.expert.cost_huber_beta = 0.5
    conf.expert.wm_coverage_paths = []
    conf.expert.wm_coverage_format = "npz"
    conf.expert.wm_coverage_max_episodes = 0
    conf.expert.wm_coverage_validation_ratio = 0.05
    _merge_dict_into_cfg(conf.expert, raw.get("expert"))

    conf.replay = CN(new_allowed=True)
    conf.replay.store_source = True
    conf.replay.store_cost = True
    conf.replay.store_safety_margin = True
    conf.replay.store_uncertainty_placeholder = True
    _merge_dict_into_cfg(conf.replay, raw.get("replay"))

    conf.logging = CN(new_allowed=True)
    conf.logging.log_expert_init_metrics = True
    _merge_dict_into_cfg(conf.logging, raw.get("logging"))

    conf.freeze()
    return conf


def cfg_to_dict(node):
    if hasattr(node, "items"):
        return {key: cfg_to_dict(value) for key, value in node.items()}
    return node
