import torch

SAMPLER_NAME = "Anima Lightning"
LEGACY_SAMPLER_NAME = "Anima Lightning (4 steps)"


def official_sigmas(steps=4):
    if isinstance(steps, bool) or int(steps) != steps or steps <= 0:
        raise ValueError("Anima Lightning sampling steps must be a positive integer")
    # Already shifted once; predictor.timestep() does not apply another shift.
    raw = torch.linspace(1.0, 0.0, int(steps) + 1, dtype=torch.float32)
    return 3 * raw / (1 + 2 * raw)


def register():
    from modules import script_callbacks, sd_samplers, sd_samplers_common
    from modules.sd_samplers_kdiffusion import KDiffusionSampler
    from .runtime import is_lightning

    class LightningSampler(KDiffusionSampler):
        def __init__(self, model):
            super().__init__("sample_euler", model)
            # Keep standard Forge CFG/callback/conditioning paths, with Euler's
            # deterministic defaults rather than user-wide churn settings.
            self.extra_params = []

        def get_sigmas(self, p, steps):
            if not is_lightning(p.sd_model):
                raise RuntimeError("Anima Lightning sampler requires the converted Anima-Lightning checkpoint")
            if self.model_wrap.predictor.multiplier != 1.0:
                raise ValueError("Anima-Lightning requires Anima's timestep multiplier of 1")
            p.extra_generation_params["Anima Lightning"] = "Euler, schedule shift=3"
            # Forge may extend the grid before cropping it for img2img/Hires.
            # Keep that native denoise-strength behavior; K follows the UI steps.
            return official_sigmas(steps)

    if SAMPLER_NAME not in sd_samplers.all_samplers_map:
        sd_samplers.add_sampler(sd_samplers_common.SamplerData(SAMPLER_NAME, LightningSampler, ["anima_lightning", LEGACY_SAMPLER_NAME], {}))
    def register_legacy_alias():
        # Other extensions rebuild this map when adding their own samplers.
        sd_samplers.all_samplers_map[LEGACY_SAMPLER_NAME] = sd_samplers.all_samplers_map[SAMPLER_NAME]
        sd_samplers.get_sampler_and_scheduler.cache_clear()

    register_legacy_alias()
    script_callbacks.on_before_ui(register_legacy_alias)
