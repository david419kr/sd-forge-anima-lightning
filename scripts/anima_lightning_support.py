import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import scripts
from anima_lightning.runtime import configure_sampling, install
from anima_lightning.sampler import register

install()
register()


class Script(scripts.Script):
    def title(self):
        return "Anima Lightning"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process_before_every_sampling(self, p, *args, **kwargs):
        configure_sampling(p)
