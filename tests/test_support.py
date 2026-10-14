import ast
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(FORGE), str(FORGE / "modules_forge/packages")]

import torch
from torch import nn
from diffusers.models.transformers.transformer_cosmos import CosmosEmbedding
from safetensors import safe_open
from anima_lightning.convert import convert, native_key
from anima_lightning.runtime import attach_k_branch, configure_sampling, is_lightning, k_condition
from anima_lightning.sampler import official_sigmas
from backend.nn.anima import Anima, LLMAdapter, Timesteps, TimestepEmbedding


class SupportTests(unittest.TestCase):
    def test_k_branch_matches_diffusers_with_optional_negpip_wrapper(self):
        torch.manual_seed(17)
        for dtype, num_steps in product((torch.float32, torch.bfloat16), (4, 6, 8)):
            reference = CosmosEmbedding(64, 64).to(dtype=dtype)
            k_reference = CosmosEmbedding(64, 64).to(dtype=dtype)
            model = nn.Module()
            model.t_embedder = nn.Sequential(Timesteps(64), TimestepEmbedding(64, 64))
            model.t_embedding_norm = nn.RMSNorm(64, eps=1e-6)
            attach_k_branch(model, TimestepEmbedding)
            model = model.to(dtype=dtype)
            engine = SimpleNamespace(forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=model))))
            configure_sampling(SimpleNamespace(sd_model=engine, steps=num_steps, is_hr_pass=False, extra_generation_params={}))
            model.t_embedder[1].linear_1.load_state_dict(reference.t_embedder.linear_1.state_dict())
            model.t_embedder[1].linear_2.load_state_dict(reference.t_embedder.linear_2.state_dict())
            model.t_embedding_norm.weight.data.copy_(reference.norm.weight)
            model.t_embedder[1].lightning_k.linear_1.load_state_dict(k_reference.t_embedder.linear_1.state_dict())
            model.t_embedder[1].lightning_k.linear_2.load_state_dict(k_reference.t_embedder.linear_2.state_dict())
            model.t_embedding_norm.lightning_k_norm.weight.data.copy_(k_reference.norm.weight)
            for layer in (model.t_embedder[1], model.t_embedding_norm):
                layer.lightning_k_scale.data.fill_(0.3)
            timesteps = torch.tensor([1.0, 0.9, 0.75, 0.5])
            hidden = torch.zeros(4, 1, 64, dtype=dtype)
            expected_t, expected_norm = reference(hidden, timesteps)
            kt, kn = k_reference(hidden, torch.full_like(timesteps, 3.0 / (num_steps + 2.0)))
            expected_t += torch.tensor(0.3, dtype=dtype) * kt
            expected_norm += torch.tensor(0.3, dtype=dtype) * kn

            def original_forward(*args, **kwargs):
                sample, temb = model.t_embedder(timesteps[:, None])
                return temb[:, 0], model.t_embedding_norm(sample.to(dtype))[:, 0]

            # Execute the installed NegPiP's actual forwarding wrapper, without
            # importing its Gradio UI or installing global attention hooks.
            source = FORGE / "extensions/sd-webui-negpip/scripts/negpip.py"
            wrapped = original_forward
            if source.is_file():
                tree = ast.parse(source.read_text(encoding="utf-8"))
                node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hook_forward_anima_model"), None)
                if node is not None:
                    scope = {"torch": torch}
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
                    wrapped = scope["hook_forward_anima_model"](None, model, original_forward)
            # Native forward casts sinusoidal embeddings to activation dtype.
            model.t_embedder[0].register_forward_hook(lambda module, args, out: out.to(dtype))
            actual_t, actual_norm = wrapped(hidden, timesteps, torch.zeros(4, 3, 64, dtype=dtype), c_negpip_mask=torch.ones(4, 3, 1))
            tolerance = 0.016 if dtype == torch.bfloat16 else 1e-5
            torch.testing.assert_close(actual_t, expected_t, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(actual_norm, expected_norm, rtol=tolerance, atol=tolerance)

    def test_per_pass_k_without_overriding_user_settings(self):
        model = SimpleNamespace(_anima_lightning=True, t_embedder=[None, SimpleNamespace()], t_embedding_norm=SimpleNamespace())
        engine = SimpleNamespace(forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=model))))
        p = SimpleNamespace(sd_model=engine, steps=6, cfg_scale=2, distilled_cfg_scale=5,
                            sampler_name="Euler", scheduler="Karras", is_hr_pass=False,
                            hr_second_pass_steps=8, hr_cfg=3, hr_distilled_cfg=4,
                            extra_generation_params={})
        before = {key: value for key, value in vars(p).items() if key != "extra_generation_params"}
        configure_sampling(p)
        self.assertEqual(model.t_embedder[1].lightning_num_steps, 6)
        self.assertEqual(model.t_embedding_norm.lightning_num_steps, 6)
        self.assertEqual(p.extra_generation_params["Anima Lightning K"], 6)
        self.assertEqual(before, {key: value for key, value in vars(p).items() if key != "extra_generation_params"})
        p.is_hr_pass = True
        configure_sampling(p)
        self.assertEqual(model.t_embedder[1].lightning_num_steps, 8)
        self.assertEqual(p.extra_generation_params["Anima Lightning Hires K"], 8)
        p.hr_second_pass_steps = 0
        configure_sampling(p)
        self.assertEqual(model.t_embedder[1].lightning_num_steps, 6)
        p.is_hr_pass = False
        p.steps = 4
        configure_sampling(p)
        self.assertEqual(model.t_embedder[1].lightning_num_steps, 4)
        normal = SimpleNamespace(sd_model=SimpleNamespace(), steps=8, extra_generation_params={})
        configure_sampling(normal)
        self.assertEqual(normal.extra_generation_params, {})

    def test_converted_model_inventory_matches_native_modules(self):
        checkpoint = FORGE / "models/Stable-diffusion/Anima-Lightning-K4-shift3-forge.safetensors"
        if not checkpoint.is_file():
            self.skipTest("Convert the supported checkpoint to run the tensor inventory check")
        with torch.device("meta"):
            model = Anima(16, 16, 2, 1, model_channels=2048, num_blocks=28, num_heads=16)
            attach_k_branch(model, TimestepEmbedding)
            adapter = LLMAdapter(1024, 1024, 1024, num_layers=6)
        expected = {"net." + k: tuple(v.shape) for k, v in model.state_dict().items()}
        expected.update({"net.llm_adapter." + k: tuple(v.shape) for k, v in adapter.state_dict().items()})
        with safe_open(str(checkpoint), framework="pt") as file:
            actual = {key: tuple(file.get_slice(key).get_shape()) for key in file.keys()}
        self.assertEqual(expected, actual)
        self.assertTrue(is_lightning(SimpleNamespace(forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=model))))))
        self.assertFalse(is_lightning(SimpleNamespace()))

    def test_schedule_and_rejected_unknown_tensors(self):
        raw = torch.tensor([1, 0.75, 0.5, 0.25, 0])
        torch.testing.assert_close(official_sigmas(), 3 * raw / (1 + 2 * raw))
        self.assertEqual(len(official_sigmas(8)), 9)
        self.assertEqual(float(official_sigmas(8)[-1]), 0.0)
        for steps in (4, 6, 8):
            self.assertEqual(len(official_sigmas(steps)), steps + 1)
            self.assertAlmostEqual(float(official_sigmas(steps)[-2]), 3 / (steps + 2), places=6)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                k_condition(torch.zeros(1, 1, 64), invalid)
            with self.assertRaises(ValueError):
                official_sigmas(invalid)
        self.assertEqual(native_key("transformer_blocks.27.attn2.to_out.0.weight"), "blocks.27.cross_attn.output_proj.weight")
        with self.assertRaises(ValueError):
            native_key("unknown.weight")
        with self.assertRaises(ValueError):
            convert({"unknown.weight": torch.zeros(1)}, {})


if __name__ == "__main__":
    unittest.main()
