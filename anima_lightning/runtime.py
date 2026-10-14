from contextvars import ContextVar
from functools import wraps
import logging
import math

import torch
from torch import nn

log = logging.getLogger("anima-lightning")
_loading = ContextVar("anima_lightning_loading", default=False)
K_KEYS = {
    "t_embedder.1.lightning_k.linear_1.weight",
    "t_embedder.1.lightning_k.linear_2.weight",
    "t_embedder.1.lightning_k_scale",
    "t_embedding_norm.lightning_k_norm.weight",
    "t_embedding_norm.lightning_k_scale",
}


def k_condition(sample, num_steps=4):
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("Anima-Lightning K must be a positive integer")
    # The trained branch uses shift=3 independently of the selected sampler.
    condition = 3.0 / (num_steps + 2.0)
    half = sample.shape[-1] // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=sample.device, dtype=torch.float32) / half)
    phases = condition * frequencies
    return torch.cat((phases.cos(), phases.sin())).reshape(1, 1, -1).to(sample)


def add_k_projection(module, inputs, output):
    sample, temb = output
    _, k_temb = module.lightning_k(k_condition(sample, module.lightning_num_steps))
    return sample, temb + module.lightning_k_scale.to(temb) * k_temb


def add_k_norm(module, inputs, output):
    return output + module.lightning_k_scale.to(output) * module.lightning_k_norm(k_condition(output, module.lightning_num_steps))


def attach_k_branch(model, embedding_type):
    channels = model.t_embedding_norm.weight.numel()
    projection = model.t_embedder[1]
    projection.lightning_num_steps = 4
    projection.lightning_k = embedding_type(channels, channels)
    projection.register_parameter("lightning_k_scale", nn.Parameter(torch.empty(())))
    norm = model.t_embedding_norm
    norm.lightning_num_steps = 4
    norm.lightning_k_norm = nn.RMSNorm(channels, eps=1e-6)
    norm.register_parameter("lightning_k_scale", nn.Parameter(torch.empty(())))
    projection.register_forward_hook(add_k_projection)
    norm.register_forward_hook(add_k_norm)
    model._anima_lightning = True


def diffusion_model(engine):
    objects = getattr(engine, "forge_objects", None)
    unet = getattr(objects, "unet", None)
    return getattr(getattr(unet, "model", None), "diffusion_model", None)


def is_lightning(engine):
    return bool(getattr(diffusion_model(engine), "_anima_lightning", False))


def configure_sampling(p):
    model = diffusion_model(p.sd_model)
    if not getattr(model, "_anima_lightning", False):
        return
    hires = bool(getattr(p, "is_hr_pass", False))
    steps = (p.hr_second_pass_steps or p.steps) if hires else p.steps
    if isinstance(steps, bool) or int(steps) != steps or steps <= 0:
        raise ValueError("Anima-Lightning sampling steps must be a positive integer")
    steps = int(steps)
    model.t_embedder[1].lightning_num_steps = steps
    model.t_embedding_norm.lightning_num_steps = steps
    key = "Anima Lightning Hires K" if hires else "Anima Lightning K"
    p.extra_generation_params[key] = steps


def install():
    from backend import loader
    from backend.logging import setup_logger
    from backend.nn.anima import Anima, TimestepEmbedding

    setup_logger(log)
    if getattr(loader.load_huggingface_component, "_anima_lightning", False):
        return
    original_init = Anima.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if _loading.get():
            attach_k_branch(self, TimestepEmbedding)

    original_load = loader.load_huggingface_component

    @wraps(original_load)
    def load(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        active = (
            component_name == "transformer"
            and cls_name == "CosmosTransformer3DModel"
            and isinstance(state_dict, dict)
            and any(key in state_dict for key in K_KEYS)
        )
        if active and not K_KEYS.issubset(state_dict):
            raise RuntimeError(f"Incomplete Anima-Lightning K branch: {sorted(K_KEYS - state_dict.keys())}")
        token = _loading.set(active)
        try:
            model = original_load(guess, component_name, lib_name, cls_name, repo_path, state_dict)
        finally:
            _loading.reset(token)
        if active:
            if not getattr(model, "_anima_lightning", False):
                raise RuntimeError("Anima-Lightning K branch was not installed by the model loader")
            log.info("Loaded Anima-Lightning with the learned variable-K branch")
        return model

    load._anima_lightning = True
    Anima.__init__ = init
    loader.load_huggingface_component = load
