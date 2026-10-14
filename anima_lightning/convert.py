"""Strict, lossless key conversion; no remotely downloaded Python is executed."""
import re

from .runtime import K_KEYS

SIMPLE = {
    "patch_embed.proj.weight": "x_embedder.proj.1.weight",
    "proj_out.weight": "final_layer.linear.weight",
    "norm_out.linear_1.weight": "final_layer.adaln_modulation.1.weight",
    "norm_out.linear_2.weight": "final_layer.adaln_modulation.2.weight",
    "time_embed.t_embedder.linear_1.weight": "t_embedder.1.linear_1.weight",
    "time_embed.t_embedder.linear_2.weight": "t_embedder.1.linear_2.weight",
    "time_embed.norm.weight": "t_embedding_norm.weight",
    "time_embed.k_embed.t_embedder.linear_1.weight": "t_embedder.1.lightning_k.linear_1.weight",
    "time_embed.k_embed.t_embedder.linear_2.weight": "t_embedder.1.lightning_k.linear_2.weight",
    "time_embed.k_embed.norm.weight": "t_embedding_norm.lightning_k_norm.weight",
    "time_embed.k_scale": "t_embedder.1.lightning_k_scale",
}


def native_key(key):
    if key in SIMPLE:
        return SIMPLE[key]
    match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", key)
    if not match:
        raise ValueError(f"Unknown transformer tensor: {key}")
    block, suffix = match.groups()
    attention = re.fullmatch(r"attn([12])\.(to_q|to_k|to_v|to_out\.0|norm_q|norm_k)\.weight", suffix)
    if attention:
        kind, name = attention.groups()
        name = {"to_q": "q_proj", "to_k": "k_proj", "to_v": "v_proj", "to_out.0": "output_proj", "norm_q": "q_norm", "norm_k": "k_norm"}[name]
        suffix = f"{'self_attn' if kind == '1' else 'cross_attn'}.{name}.weight"
    elif suffix in ("ff.net.0.proj.weight", "ff.net.2.weight"):
        suffix = "mlp.layer1.weight" if suffix == "ff.net.0.proj.weight" else "mlp.layer2.weight"
    else:
        norm = re.fullmatch(r"norm([123])\.linear_([12])\.weight", suffix)
        if not norm:
            raise ValueError(f"Unknown transformer block tensor: {key}")
        kind, linear = norm.groups()
        suffix = f"adaln_modulation_{dict(zip('123', ['self_attn', 'cross_attn', 'mlp']))[kind]}.{linear}.weight"
    return f"blocks.{block}.{suffix}"


def convert(transformer, conditioner):
    result = {}
    for key, tensor in transformer.items():
        destination = "net." + native_key(key)
        if destination in result:
            raise ValueError(f"Duplicate target tensor: {destination}")
        result[destination] = tensor
    result["net.t_embedding_norm.lightning_k_scale"] = transformer["time_embed.k_scale"].clone()
    if not all("net." + key in result for key in K_KEYS):
        raise ValueError("Missing trained Lightning K tensors")
    for key, tensor in conditioner.items():
        result["net.llm_adapter." + key] = tensor
    return result
