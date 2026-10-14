import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from anima_lightning.convert import convert

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
if args.destination.exists():
    raise FileExistsError(args.destination)
config = json.loads((args.source / "transformer/config.json").read_text())
if config.get("tdm_unify_shift") != 3.0 or config.get("_class_name") != "CosmosTDMUnifyTransformer3DModel":
    raise ValueError("Expected the current shift-3 TDM-Unify Anima-Lightning release")
with (args.source / "transformer/diffusion_pytorch_model.safetensors").open("rb") as source_file:
    if hashlib.file_digest(source_file, "sha256").hexdigest() != "8ea644535ab333540320c6c3d13969c4487c1a6f1135898322317b0b0fb4bf99":
        raise ValueError("Transformer does not match the documented source revision; review before converting")
transformer = load_file(str(args.source / "transformer/diffusion_pytorch_model.safetensors"))
conditioner = load_file(str(args.source / "text_conditioner/diffusion_pytorch_model.safetensors"))
converted = convert(transformer, conditioner)
temporary = args.destination.with_suffix(".safetensors.partial")
save_file(converted, str(temporary), metadata={
    "modelspec.title": "Anima Lightning (K=4, shift=3)",
    "anima_lightning": "tdm-unify-k4-shift3",
    "source": "https://huggingface.co/aina-tech/Anima-Lightning",
    "source_revision": "9dbb6e1a0f94eda0a1e523836d32355fb961cc7a",
    "license": "circlestone-labs-non-commercial-license",
})
with safe_open(str(temporary), framework="pt") as check:
    if set(check.keys()) != set(converted):
        raise ValueError("Converted tensor inventory mismatch")
temporary.rename(args.destination)
print(f"Saved {len(converted)} tensors to {args.destination} ({args.destination.stat().st_size:,} bytes)")
