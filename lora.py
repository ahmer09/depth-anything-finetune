
"""
LoRA (Low-Rank Adaptation) Module
==================================
Self-contained implementation — no external dependencies needed.

How LoRA works:
    Instead of updating a large weight matrix W (e.g., 384×384 = 147K params),
    LoRA freezes W and adds a tiny bypass: W + A×B

    Original:   y = x @ W                     (W is frozen, 0 trainable params)
    With LoRA:  y = x @ W + x @ A @ B × scale (A: 384×8, B: 8×384 = 6K trainable!)

    - A is initialised randomly (Kaiming)
    - B is initialised to zeros → LoRA starts as identity (no change)
    - rank r controls the capacity (4-16 typical)
    - alpha/r is the scaling factor

Where LoRA is applied:
    In Vision Transformers, each attention block has:
        Q, K, V projections  (qkv linear layer)
        Output projection     (proj linear layer)
        MLP layers            (fc1, fc2)

    We inject LoRA into the attention qkv and proj layers,
    which are the most impactful for feature adaptation.

Usage:
    from lora import apply_lora, get_lora_params, save_lora, load_lora

    model = DepthAnythingV2(...)
    model.load_state_dict(...)  # load pre-trained weights

    # Apply LoRA to encoder attention layers
    apply_lora(model.pretrained, rank=8, alpha=16, target_modules=["qkv", "proj"])

    # Only train LoRA params + decoder
    optimizer = AdamW([
        {"params": get_lora_params(model), "lr": 1e-4},
        {"params": decoder_params, "lr": 1e-3},
    ])

    # After training, save just the LoRA weights (tiny file!)
    save_lora(model, "lora_nyu_r8.pth")  # ~200KB vs 100MB full model

    # Load LoRA onto a fresh model later
    load_lora(model, "lora_nyu_r8.pth")
"""

import torch
import torch.nn as nn
import math


# ========================================================================== #
#                           LORA LINEAR LAYER                                #
# ========================================================================== #

class LoRALinear(nn.Module):
    """
    A wrapper that adds LoRA adapters to an existing nn.Linear layer.

    Original computation:  y = x @ W + bias
    With LoRA:             y = x @ W + bias + (x @ A @ B) * scaling

    The original weight W is FROZEN. Only A and B are trainable.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()

        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # LoRA matrices
        # A: down-projection (in_features → rank)
        # B: up-projection   (rank → out_features)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # Initialise A with Kaiming, B with zeros
        # This means LoRA starts as identity (A×B = 0 initially)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already zeros

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        # Original frozen computation
        original_out = self.original(x)

        # LoRA bypass: x → dropout → A → B → scale
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling

        return original_out + lora_out

    def extra_repr(self):
        return (f"in={self.original.in_features}, out={self.original.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.2f}")


# ========================================================================== #
#                          APPLY / REMOVE LORA                               #
# ========================================================================== #

def apply_lora(model, rank=8, alpha=16.0, dropout=0.0, target_modules=None):
    """
    Walk through the model and replace matching nn.Linear layers with LoRALinear.

    Args:
        model:          The model (or submodule like model.pretrained) to modify
        rank:           LoRA rank (4-16 typical, higher = more capacity)
        alpha:          LoRA scaling factor (usually 2× rank)
        dropout:        Dropout on LoRA path (0.0-0.1)
        target_modules: List of layer name substrings to target.
                        Default: ["qkv", "proj"] (attention layers)

    Returns:
        Number of LoRA layers injected
    """
    if target_modules is None:
        target_modules = ["qkv", "proj"]

    count = 0
    replacements = []

    # Find all linear layers matching target names
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer's name matches any target
            if any(target in name for target in target_modules):
                replacements.append((name, module))

    # Apply replacements
    for name, module in replacements:
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace with LoRA version
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_layer)
        count += 1

    return count


def get_lora_params(model):
    """Get only the LoRA trainable parameters (A and B matrices)."""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
    return lora_params


def count_lora_params(model):
    """Count total LoRA parameters."""
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
    return total, trainable


# ========================================================================== #
#                         SAVE / LOAD LORA WEIGHTS                           #
# ========================================================================== #

def save_lora(model, path):
    """
    Save ONLY the LoRA weights to a file.
    This produces a tiny checkpoint (~200KB for rank=8 on ViT-S).
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()

    metadata = {
        "lora_state_dict": lora_state,
        "num_lora_params": sum(p.numel() for p in lora_state.values()),
    }
    torch.save(metadata, path)
    return len(lora_state)


def load_lora(model, path):
    """
    Load LoRA weights into a model that already has LoRA layers applied.
    """
    checkpoint = torch.load(path, map_location="cpu")
    lora_state = checkpoint["lora_state_dict"]

    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name] = param

    model.load_state_dict(model_state, strict=False)
    return len(lora_state)


# ========================================================================== #
#                            MERGE LORA                                      #
# ========================================================================== #

def merge_lora(model):
    """
    Merge LoRA weights INTO the original weights permanently.

    After merging:
        W_new = W_original + (A @ B) * scaling

    This eliminates the runtime overhead of LoRA (no extra computation)
    while keeping the adapted behaviour. Useful for deployment.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Compute the LoRA delta: A @ B * scaling
            with torch.no_grad():
                delta = (module.lora_A @ module.lora_B) * module.scaling
                module.original.weight.data += delta.T  # Linear stores weight transposed

            # Replace LoRALinear with the original (now modified) Linear
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module.original)
            count += 1

    return count


# ========================================================================== #
#                            SUMMARY                                         #
# ========================================================================== #

def lora_summary(model):
    """Print a summary of LoRA layers in the model."""
    print("\n  LoRA Summary:")
    print(f"  {'Layer':<50} {'Shape A':<15} {'Shape B':<15} {'Params':<10}")
    print(f"  {'-'*90}")

    total = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_shape = tuple(module.lora_A.shape)
            b_shape = tuple(module.lora_B.shape)
            params = module.lora_A.numel() + module.lora_B.numel()
            total += params
            print(f"  {name:<50} {str(a_shape):<15} {str(b_shape):<15} {params:<10}")

    print(f"  {'-'*90}")
    print(f"  {'Total LoRA params':<50} {'':<15} {'':<15} {total:<10}")
    print(f"  {'Size on disk':<50} {'':<15} {'':<15} {f'~{total * 4 / 1024:.0f} KB':<10}")
    print()
