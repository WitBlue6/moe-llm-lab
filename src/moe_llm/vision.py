"""Native frozen SigLIP + projector + explicitly enabled visual LoRA.

The fixture backend is a random frozen patch convolution, solely for offline
engineering tests. It is not a pretrained or useful vision encoder.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .lora import inject_visual_lora
from .model import LanguageModel, ModelConfig
from .tokenizer import sha256_file


@dataclass(frozen=True)
class VisionConfig:
    encoder_type: str = "siglip"
    encoder_path: str = "models/siglip-base-patch16-224"
    image_token_grid: int = 8
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_targets: tuple = ("q_proj", "v_proj")
    fixture_image_size: int = 16
    fixture_patch_size: int = 4
    fixture_hidden_size: int = 16
    fixture_seed: int = 1234

    def __post_init__(self):
        object.__setattr__(self, "lora_targets", tuple(self.lora_targets))
        if self.encoder_type not in ("siglip", "fixture"):
            raise ValueError("encoder_type must be siglip or fixture")
        if min(self.image_token_grid, self.lora_rank, self.fixture_image_size,
               self.fixture_patch_size, self.fixture_hidden_size) <= 0 or self.lora_alpha <= 0:
            raise ValueError("vision dimensions and LoRA settings must be positive")
        if self.fixture_image_size % self.fixture_patch_size:
            raise ValueError("fixture image size must be divisible by patch size")
        if not self.lora_targets or len(set(self.lora_targets)) != len(self.lora_targets) or not set(self.lora_targets) <= {"q_proj", "k_proj", "v_proj", "o_proj"}:
            raise ValueError("invalid LoRA targets")

    @property
    def image_tokens(self):
        return self.image_token_grid ** 2

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def encoder_fingerprint(config):
    if config.encoder_type == "fixture":
        return {"kind": "synthetic_random_fixture", "image_size": config.fixture_image_size,
                "patch_size": config.fixture_patch_size, "hidden_size": config.fixture_hidden_size,
                "seed": config.fixture_seed}
    root = Path(config.encoder_path)
    files = [root / "config.json", root / "preprocessor_config.json"]
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors weights in {root}; download a local SigLIP snapshot first")
    files += weights + sorted(root.glob("*.safetensors.index.json"))
    return {path.name: sha256_file(path) for path in files}


class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.backend = None
        if config.encoder_type == "siglip":
            from .siglip import SiglipConfig, NativeImageProcessor
            root = Path(config.encoder_path)
            c = SiglipConfig.load(root / 'config.json')
            self.backend = NativeImageProcessor(c.image_size, json.loads((root / 'preprocessor_config.json').read_text()))

    def __call__(self, path):
        from PIL import Image, ImageOps
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if self.backend is not None:
                return self.backend(image)
            import numpy as np
            size = self.config.fixture_image_size
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            array = np.array(image, dtype=np.float32, copy=True) / 127.5 - 1
            return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class FrozenVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.encoder_type == "siglip":
            from .siglip import NativeSiglipVision
            self.encoder = NativeSiglipVision.from_local(config.encoder_path)
            self.hidden_size = self.encoder.config.hidden_size
            self.image_size = self.encoder.config.image_size
            patch_size = self.encoder.config.patch_size
        else:
            # Reproducible synthetic weights without disturbing the caller's RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.fixture_seed)
                self.encoder = nn.Conv2d(3, config.fixture_hidden_size,
                                         config.fixture_patch_size, config.fixture_patch_size, bias=False)
            self.hidden_size = config.fixture_hidden_size
            self.image_size = config.fixture_image_size
            patch_size = config.fixture_patch_size
        if self.image_size // patch_size < config.image_token_grid:
            raise ValueError("image_token_grid must not upsample the encoder's patch grid")
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixel_values):
        if pixel_values.ndim != 4 or pixel_values.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("pixel_values shape differs from the encoder image size")
        if self.config.encoder_type == "siglip":
            features = self.encoder(pixel_values=pixel_values)
            grid = math.isqrt(features.shape[1])
            if grid * grid != features.shape[1]:
                raise ValueError("expected a square SigLIP patch grid without CLS token")
            features = features.transpose(1, 2).reshape(features.shape[0], self.hidden_size, grid, grid)
        else:
            features = self.encoder(pixel_values)
        return F.adaptive_avg_pool2d(features, self.config.image_token_grid).flatten(2).transpose(1, 2)


class VisionLanguageModel(nn.Module):
    def __init__(self, language_model, vision_config):
        super().__init__()
        self.config = language_model.config
        self.vision_config = vision_config
        self.language_model = language_model.requires_grad_(False)
        self.vision_encoder = FrozenVisionEncoder(vision_config)
        self.projector = nn.Sequential(nn.LayerNorm(self.vision_encoder.hidden_size),
                                       nn.Linear(self.vision_encoder.hidden_size, self.config.hidden_size),
                                       nn.GELU(), nn.Linear(self.config.hidden_size, self.config.hidden_size))
        inject_visual_lora(language_model, vision_config.lora_rank,
                           vision_config.lora_alpha, vision_config.lora_targets)
        self.set_stage("align")

    def set_stage(self, stage):
        if stage not in ("align", "sft"):
            raise ValueError("visual training stage must be align or sft")
        self.requires_grad_(False)
        self.projector.requires_grad_(True)
        if stage == "sft":
            for name, parameter in self.language_model.named_parameters():
                if name.endswith(("lora_A", "lora_B")):
                    parameter.requires_grad_(True)
        self.stage = stage

    def adapter_state_dict(self):
        return {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()
                if name.startswith("projector.") or name.endswith(("lora_A", "lora_B"))}

    def load_adapter_state_dict(self, state):
        expected = self.adapter_state_dict()
        if set(state) != set(expected):
            raise ValueError("visual checkpoint adapter keys mismatch")
        for name in expected:
            if state[name].shape != expected[name].shape:
                raise ValueError(f"adapter shape mismatch: {name}")
        self.load_state_dict(state, strict=False)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_positions=None,
                past_key_values=None, use_cache=False, visual_context=False, return_hidden=False):
        image_present = pixel_values is not None
        if image_present and past_key_values is not None:
            raise ValueError("images are encoded only during prefill; cache already contains them")
        if image_positions is not None and not image_present:
            raise ValueError("image positions require pixel_values")
        if visual_context and past_key_values is None and not image_present:
            raise ValueError("visual_context without image requires an existing visual cache")
        if image_present:
            batch, length = input_ids.shape
            if pixel_values.shape[0] != batch or image_positions is None or image_positions.shape != (batch,):
                raise ValueError("one image and one insertion position are required per sequence")
            count = self.vision_config.image_tokens
            embeds = self.language_model.embedding(input_ids)
            visual = self.projector(self.vision_encoder(pixel_values).to(embeds.dtype))
            rows = []
            for row in range(batch):
                position = int(image_positions[row])
                if position < 0 or position + count > length:
                    raise ValueError("image span exceeds the input sequence")
                if not torch.all(input_ids[row, position:position + count] == 0):
                    raise ValueError("image slots must contain placeholder ID 0")
                if attention_mask is not None and not torch.all(attention_mask[row, position:position + count]):
                    raise ValueError("visual positions must be visible in the attention mask")
                rows.append(torch.cat((embeds[row, :position], visual[row], embeds[row, position + count:]), 0))
            # The frozen LLM must retain autograd here so loss reaches the projector.
            return self.language_model(inputs_embeds=torch.stack(rows), attention_mask=attention_mask,
                use_cache=use_cache, return_hidden=return_hidden, adapter_enabled=True)
        return self.language_model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, return_hidden=return_hidden, adapter_enabled=visual_context)


@torch.inference_mode()
def generate_visual(model, prompt_ids, pixel_values=None, image_positions=None,
                    max_new_tokens=64, temperature=0.8, top_k=40, eos_id=2):
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] >= model.config.max_seq_len:
        raise ValueError("provide one unpadded prompt with room for a response")
    if max_new_tokens < 0 or temperature < 0 or top_k < 0:
        raise ValueError("invalid generation options")
    model.eval()
    ids, current, cache = prompt_ids, prompt_ids, None
    with_image = pixel_values is not None
    for _ in range(min(max_new_tokens, model.config.max_seq_len - ids.shape[1])):
        out = model(current, pixel_values=pixel_values if cache is None else None,
                    image_positions=image_positions if cache is None else None,
                    past_key_values=cache, use_cache=True, visual_context=with_image and cache is not None)
        logits = out["logits"][:, -1].float()
        # Image placeholders use existing ID 0; no new tokens or output classes.
        if temperature == 0:
            next_id = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, -float("inf"))
            next_id = torch.multinomial(logits.softmax(-1), 1)
        ids = torch.cat((ids, next_id), 1)
        if next_id.item() == eos_id:
            break
        current, cache = next_id, out["past_key_values"]
    return ids
