"""Native PyTorch SigLIP vision backbone and a small sigmoid-contrastive learner.

The vision patch-token path is compatible with first-generation HF SigLIP
weights. Scratch training is a deliberately smaller SigLIP-style experiment:
mean image pooling, EOS text pooling, learned projection, scale and bias. It is
not a reproduction of Google's data, training recipe, or released performance.
Reference: https://arxiv.org/abs/2303.15343
"""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SiglipConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    attention_dropout: float = 0.0

    def __post_init__(self):
        if min(self.image_size, self.patch_size, self.hidden_size, self.intermediate_size,
               self.num_hidden_layers, self.num_attention_heads) <= 0:
            raise ValueError("SigLIP dimensions must be positive")
        if self.image_size % self.patch_size or self.hidden_size % self.num_attention_heads:
            raise ValueError("image/patch and hidden/head sizes must divide exactly")
        if self.num_channels != 3 or self.hidden_act not in ("gelu", "gelu_pytorch_tanh"):
            raise ValueError("only RGB and GELU SigLIP backbones are supported")
        if not 0 <= self.attention_dropout < 1 or self.layer_norm_eps <= 0:
            raise ValueError("invalid dropout or normalization epsilon")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('model_type', 'siglip') not in ('siglip', 'siglip_vision_model', 'moe-lab-siglip-vision'):
            raise ValueError("expected first-generation SigLIP or this project's export; SigLIP2 is not supported")
        model_type = data.get('model_type', 'siglip')
        data = data.get('vision_config', data)
        if model_type in ('siglip', 'siglip_vision_model'):
            # Official configs omit fields equal to HF SiglipVisionConfig defaults.
            defaults = asdict(cls(hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12))
            data = {**defaults, **data}
        names = {f.name for f in fields(cls)}
        required = ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads')
        if any(k not in data for k in required):
            raise ValueError("incomplete vision configuration")
        return cls(**{k: v for k, v in data.items() if k in names})


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_attention_heads
        self.dropout = c.attention_dropout
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            setattr(self, name, nn.Linear(c.hidden_size, c.hidden_size))

    def forward(self, x, mask=None):
        b, n, d = x.shape
        q, k, v = [getattr(self, name)(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
                   for name in ('q_proj', 'k_proj', 'v_proj')]
        y = F.scaled_dot_product_attention(q, k, v,
            attn_mask=None if mask is None else mask[:, None, None, :].bool(),
            dropout_p=self.dropout if self.training else 0.)
        return self.out_proj(y.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.fc2 = nn.Linear(c.intermediate_size, c.hidden_size)
        self.approximate = 'tanh' if c.hidden_act == 'gelu_pytorch_tanh' else 'none'

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate=self.approximate))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.self_attn = Attention(c)
        self.mlp = MLP(c)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList([Block(c) for _ in range(c.num_hidden_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PatchEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, c.hidden_size, c.patch_size, c.patch_size)
        self.position_embedding = nn.Embedding((c.image_size // c.patch_size) ** 2, c.hidden_size)

    def forward(self, pixels):
        x = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        return x + self.position_embedding.weight.unsqueeze(0)


class NativeSiglipVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = PatchEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        c = self.config
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (3, c.image_size, c.image_size):
            raise ValueError("pixel shape must match native vision config")
        return self.post_layernorm(self.encoder(self.embeddings(pixel_values)))

    @classmethod
    def from_local(cls, root):
        # Read only backbone tensors, never instantiate an external model or execute repo code.
        from safetensors import safe_open
        root = Path(root)
        model = cls(SiglipConfig.load(root / 'config.json'))
        expected, weights = set(model.state_dict()), {}
        files = sorted(root.glob('*.safetensors'))
        if not files:
            raise FileNotFoundError(f"no safetensors files in {root}")
        for file in files:
            with safe_open(file, framework='pt', device='cpu') as reader:
                for key in reader.keys():
                    name = key.removeprefix('vision_model.')
                    if name in expected:
                        if name in weights:
                            raise ValueError(f"duplicate vision tensor: {name}")
                        weights[name] = reader.get_tensor(key)
                    elif key.startswith(('text_model.', 'vision_model.head.', 'head.')) or key in ('logit_scale', 'logit_bias'):
                        continue  # not used by the VLM patch-token path
                    else:
                        raise ValueError(f"unsupported tensor in vision checkpoint: {key}")
        model.load_state_dict(weights, strict=True)
        return model


class NativeImageProcessor:
    def __init__(self, image_size, config=None):
        self.image_size = image_size
        c = config or {}
        if any(c.get(k, True) is not True for k in ('do_resize', 'do_rescale', 'do_normalize')):
            raise ValueError("unsupported image processing switches")
        size = c.get('size', {'height': image_size, 'width': image_size})
        if size != {'height': image_size, 'width': image_size}:
            raise ValueError("processor size differs from model config")
        if c.get('resample', 3) != 3:
            raise ValueError("native processor requires bicubic resampling")
        self.scale = c.get('rescale_factor', 1 / 255)
        self.mean = np.asarray(c.get('image_mean', [0.5] * 3), dtype=np.float32)
        self.std = np.asarray(c.get('image_std', [0.5] * 3), dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or (self.std <= 0).any():
            raise ValueError("invalid normalization")

    def __call__(self, image):
        from PIL import Image
        image = image.convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = (np.asarray(image).astype(np.float64) * self.scale).astype(np.float32)
        return torch.from_numpy((array - self.mean) / self.std).permute(2, 0, 1).contiguous()


class ContrastiveSiglip(nn.Module):
    def __init__(self, vision_config, vocab_size, text_length=128, projection_size=256, text_layers=4):
        super().__init__()
        from dataclasses import replace
        self.vision = NativeSiglipVision(vision_config)
        c = replace(vision_config, num_hidden_layers=text_layers)
        self.text_embedding = nn.Embedding(vocab_size, c.hidden_size)
        self.text_position = nn.Embedding(text_length, c.hidden_size)
        self.text_encoder = Encoder(c)
        self.text_norm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.image_projection = nn.Linear(c.hidden_size, projection_size)
        self.text_projection = nn.Linear(c.hidden_size, projection_size)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.))

    def forward(self, pixels, tokens, mask):
        image = self.image_projection(self.vision(pixels).mean(dim=1))
        x = self.text_embedding(tokens) + self.text_position.weight[:tokens.shape[1]]
        x = self.text_norm(self.text_encoder(x, mask))
        eos = mask.sum(dim=1) - 1
        text = self.text_projection(x[torch.arange(x.shape[0], device=x.device), eos])
        return F.normalize(image.float(), dim=-1), F.normalize(text.float(), dim=-1), self.logit_scale.exp().clamp(max=100), self.logit_bias


def sigmoid_pair_loss(image, text, scale, bias, positive_columns=None):
    """Sum pair losses / number of images, NOT mean over all N*N pairs."""
    logits = image.float() @ text.float().T * scale.float() + bias.float()
    columns = torch.arange(len(image), device=image.device) if positive_columns is None else positive_columns
    targets = torch.zeros_like(logits)
    targets[torch.arange(len(image), device=image.device), columns] = 1
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='sum') / len(image)
