"""
evo2_regression.py  —  RiboScanner / Evo2 integration
=======================================================
Three fine-tuning strategies for using Evo2 as a pretrained backbone
to predict Reporter RiboScan measurements from 5'UTR sequences (≤150 bp).

STRATEGIES
----------
  Strategy 1 — Frozen backbone + MLP head
      Only the regression head is trained. Fast, low memory, good baseline.
      Best when your dataset is small (<5k sequences).

  Strategy 2 — Full fine-tuning
      All Evo2 weights + head are updated. Highest ceiling but needs more
      data and a GPU with enough VRAM (evo2_7b_base needs ~20 GB in bf16).

  Strategy 3 — LoRA fine-tuning
      Low-rank adapters are injected into every linear projection of Evo2.
      Only the adapter weights + head are trained. Good balance between
      quality and memory. Requires `peft` library.

IMPORTANT — model choice
-------------------------
  evo2_7b_base  : bfloat16, no FP8 / Transformer Engine needed. ✓ Recommended.
  evo2_1b_base  : requires FP8 + Transformer Engine + Hopper GPU. ✗ Harder.

For ≤150 bp sequences, evo2_7b_base is well-suited and runs on a single A100/H100.

ARCHITECTURE NOTE
-----------------
Evo2 is a StripedHyena causal model. It operates at single-nucleotide
resolution via a character-level tokeniser (one token = one nucleotide).
150 bp → 150 tokens (tiny context). We extract the *mean* of the final-layer
hidden states as the sequence embedding and feed it into the regression head.

USAGE
-----
  # Strategy 1 — frozen backbone
  python evo2_regression.py --strategy frozen \
      --train_csv data/train.csv --val_csv data/val.csv \
      --seq_col Sequence --label_col mean_GFP \
      --output_dir outputs/evo2-frozen

  # Strategy 2 — full fine-tuning
  python evo2_regression.py --strategy full \
      --train_csv data/train.csv --val_csv data/val.csv \
      --lr 1e-5 --output_dir outputs/evo2-full

  # Strategy 3 — LoRA
  python evo2_regression.py --strategy lora \
      --lora_r 8 --lora_alpha 16 \
      --train_csv data/train.csv --val_csv data/val.csv \
      --output_dir outputs/evo2-lora

INTEGRATION WITH RIBOSCANNER
-----------------------------
Drop this file into RiboScanner/ alongside train_model.py and add
'evo2_frozen' | 'evo2_full' | 'evo2_lora' as choices in cli.py.
The Dataset class (Evo2Dataset) plugs into the existing DataLoader pipeline.
"""

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on HPC
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import seaborn as sns
import os
os.environ["HF_HOME"] = "/hpc/compgen/users/lbarbadillamartinez/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/hpc/compgen/users/lbarbadillamartinez/.cache/huggingface"
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# ── Evo2 imports ──────────────────────────────────────────────────────────────
# Evo2 must be installed: pip install evo2  (+ flash-attn for 7B)
try:
    from evo2 import Evo2
except ImportError:
    raise ImportError(
        "evo2 is not installed. Install with:\n"
        "  pip install flash-attn==2.8.0.post2 --no-build-isolation\n"
        "  pip install evo2"
    )

# ── LoRA imports (optional) ───────────────────────────────────────────────────
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# =============================================================================
# Output directory helpers
# =============================================================================

def resolve_output_dir(base_dir: str) -> str:
    """
    If *base_dir* does not exist yet, return it as-is.
    If it already exists, return  base_dir/trial_1, base_dir/trial_2, …
    choosing the first index that does not yet exist.
    """
    base = Path(base_dir)
    if not base.exists():
        return str(base)

    trial = 1
    while True:
        candidate = base / f"trial_{trial}"
        if not candidate.exists():
            return str(candidate)
        trial += 1


def save_args(args: argparse.Namespace, output_dir: str):
    """Serialise all CLI arguments to <output_dir>/args.json."""
    args_path = os.path.join(output_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  Arguments saved → {args_path}")


# =============================================================================
# Plotting helpers
# =============================================================================


def save_scatter_plot(
    preds: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    split: str,        # "train" | "val" | "test"
    output_dir: str,
    pcc: float,
    spearman: float,
):
    """
    Scatter plot of predicted vs. measured values for *split* at *epoch*.
    Saved as  <output_dir>/plots/<split>_epoch<epoch:03d>.png
    """
    import seaborn as sns
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    not_nan_pred = ~np.isnan(preds)
    preds = preds[not_nan_pred]
    labels = labels[not_nan_pred]

    #Make jointplot
    g= sns.jointplot(x=labels,y=preds, kind='hex', gridsize=40, cmap='afmhot_r', 
                    marginal_kws=dict(bins=75, fill=True, color='black'))

    g.ax_joint.hist2d(labels, preds, bins=(40, 40), norm=colors.LogNorm(), cmap='afmhot_r' )

    # Identity line spanning the data range
    lo = min(labels.min(), preds.min())
    hi = max(labels.max(), preds.max())

    #Take the ax of jointplot
    ax = g.ax_joint

    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Measured", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)

    ax.set_title(
        f"{split.capitalize()}  |  epoch {epoch}\n"
        f"PCC = {pcc:.3f}   Spearman = {spearman:.3f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    g.fig.tight_layout()

    fname = os.path.join(plots_dir, f"{split}_epoch{epoch:03d}.png")
    g.fig.savefig(fname, dpi=120)
    plt.close(g.fig)




# =============================================================================
# Label normalisation
# =============================================================================

class LabelScaler:
    """
    Z-score normaliser for regression targets.
    Fit on training labels; apply to train/val/test; inverse-transform predictions.

    Use with  --normalize_labels  to stabilise training when label values
    span a wide range (e.g. raw GFP counts in the thousands).
    """

    def __init__(self):
        self.mean = 0.0
        self.std  = 1.0

    def fit(self, labels: list):
        a = np.array(labels, dtype=float)
        self.mean = float(a.mean())
        self.std  = float(a.std()) or 1.0   # guard against zero std
        print(f"  LabelScaler fit: mean={self.mean:.4f}  std={self.std:.4f}")
        return self

    def transform(self, labels: list) -> list:
        a = np.array(labels, dtype=float)
        return ((a - self.mean) / self.std).tolist()

    def inverse_transform(self, preds: np.ndarray) -> np.ndarray:
        return preds * self.std + self.mean

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            d = json.load(f)
        s = cls()
        s.mean, s.std = d["mean"], d["std"]
        return s


# =============================================================================
# Loss-curve plot
# =============================================================================

def save_loss_curve(history: list, output_dir: str):
    """
    Saves  <output_dir>/plots/loss_curve.png  with four panels:
      - train & val loss over epochs
      - train & val PCC over epochs
      - mean gradient norm over epochs
      - prediction std over epochs (collapses to 0 when predicting constant)
    Updated after every epoch so you can inspect it mid-run.
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs      = [r["epoch"]          for r in history]
    train_loss  = [r["train_loss"]     for r in history]
    val_loss    = [r["val_loss"]       for r in history]
    train_pcc   = [r["train_pcc"]      for r in history]
    val_pcc     = [r["val_pcc"]        for r in history]
    grad_norms  = [r.get("grad_norm",  float("nan")) for r in history]
    pred_stds   = [r.get("pred_std",   float("nan")) for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Training diagnostics", fontsize=13)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train", color="#2563EB")
    ax.plot(epochs, val_loss,   label="val",   color="#DC2626", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # PCC
    ax = axes[0, 1]
    ax.plot(epochs, train_pcc, label="train", color="#2563EB")
    ax.plot(epochs, val_pcc,   label="val",   color="#DC2626", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("PCC"); ax.set_title("Pearson r")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 0]
    ax.plot(epochs, grad_norms, color="#7C3AED")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Grad norm")
    ax.set_title("Mean gradient norm\n(spikes → exploding gradients)")
    ax.grid(True, alpha=0.3)

    # Prediction std — a flat 0 here means constant-output collapse
    ax = axes[1, 1]
    ax.plot(epochs, pred_stds, color="#059669")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Std of predictions")
    ax.set_title("Prediction std (train)\n(~0 = constant-output collapse)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=120)
    plt.close(fig)


# =============================================================================
# Dataset
# =============================================================================

class Evo2Dataset(Dataset):
    """
    Dataset that tokenises raw DNA strings for Evo2.

    Evo2 uses a CharLevelTokenizer: each nucleotide → one integer token.
    No padding needed inside the dataset; we pad to the batch maximum
    in the collate function below.

    Args:
        sequences : list of DNA strings (up to 150 bp)
        labels    : list/array of float targets
        tokenizer : evo2_model.tokenizer  (CharLevelTokenizer)
    """

    def __init__(self, sequences, labels, tokenizer):
        self.sequences  = sequences
        self.labels     = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer  = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].upper().replace('U', 'T')   # DNA only
        ids = torch.tensor(self.tokenizer.tokenize(seq), dtype=torch.long)
        return ids, self.labels[idx]


def collate_fn(batch, pad_id: int):
    """
    Pads sequences in a batch to the same length (right-padding with pad_id).
    Returns:
        input_ids   : (batch, max_len)   long
        attn_mask   : (batch, max_len)   bool  (True = real token)
        labels      : (batch,)           float
    """
    ids_list, label_list = zip(*batch)
    max_len = max(x.shape[0] for x in ids_list)

    padded, masks = [], []
    for ids in ids_list:
        pad_len = max_len - ids.shape[0]
        padded.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(ids.shape[0], dtype=torch.bool),
                                torch.zeros(pad_len, dtype=torch.bool)]))

    return (
        torch.stack(padded),
        torch.stack(masks),
        torch.stack(label_list),
    )


def load_csv(path, seq_col, label_col):
    df = pd.read_csv(path)
    missing = {seq_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in {path}: {missing}")
    return df[seq_col].tolist(), df[label_col].astype(float).tolist()


# =============================================================================
# Regression head
# =============================================================================

_ACTIVATIONS = {
    "gelu":  nn.GELU,
    "relu":  nn.ReLU,
    "silu":  nn.SiLU,
    "tanh":  nn.Tanh,
    "leaky": nn.LeakyReLU,
}


class RegressionHead(nn.Module):
    """
    MLP head that takes a pooled Evo2 embedding and predicts a scalar.

    hidden_dim  — size of Evo2's hidden states
                  evo2_1b_base : 1920
                  evo2_7b_base : 4096
    mlp_dims    — list of intermediate layer widths; [] = single linear layer
    dropout     — dropout probability applied after each hidden activation
    activation  — one of 'gelu' | 'relu' | 'silu' | 'tanh' | 'leaky'
    batch_norm  — if True, adds BatchNorm1d before each activation
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_dims=(256, 64),
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_norm: bool = False,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. "
                             f"Choose from: {list(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[activation]
        layers  = []
        in_dim  = hidden_dim
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (batch,)


# =============================================================================
# Full model wrapper
# =============================================================================

HIDDEN_DIM = {
    'evo2_1b_base': 1920,
    'evo2_7b_base': 4096,
    'evo2_7b':      4096,
}

# Layer to extract embeddings from (penultimate MLP output, as recommended in paper)
EMBEDDING_LAYER = {
    'evo2_1b_base': 'blocks.22.mlp.l3',
    'evo2_7b_base': 'blocks.28.mlp.l3',
    'evo2_7b':      'blocks.28.mlp.l3',
}

class AttentionPool(nn.Module):
    """
    Learned attention pooling over the sequence dimension.
 
    Unlike mean/max pooling (which are position-invariant but treat all
    tokens equally) or per-position weighting like a bilinear gene-token head
    (which requires a fixed, meaningful position index — not applicable to
    variable-length, unanchored sequences), this layer learns WHICH KIND of
    local pattern is relevant and weights tokens accordingly, regardless of
    where they occur in the sequence.
 
    A single learned query vector is compared against every token's hidden
    state (dot-product attention, single head). The resulting scores are
    softmax-normalised over only the real (non-padded) tokens, then used to
    take a weighted sum of the token embeddings.
 
        scores  = hidden @ query / sqrt(hidden_dim)      (batch, seq_len)
        weights = softmax(scores masked to real tokens)  (batch, seq_len)
        pooled  = sum(weights * hidden, dim=seq_len)      (batch, hidden_dim)
 
    This adds hidden_dim trainable parameters (the query vector) — negligible
    compared to the head, and far smaller than a per-position bilinear head
    which would need n_genes/n_positions * rank parameters and breaks on
    variable-length input.
    """
 
    def __init__(self, hidden_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.randn(hidden_dim) * init_scale)
 
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        hidden : (batch, seq_len, hidden_dim)  float
        mask   : (batch, seq_len)              float, 1.0 = real token, 0.0 = padding
        Returns: (batch, hidden_dim)
        """
        hidden = hidden.float()
        
        scores = torch.einsum("bsd,d->bs", hidden, self.query) / (self.hidden_dim ** 0.5)
        # Mask out padding positions before softmax so they get ~0 weight
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)      # (batch, seq_len, 1)
        pooled = (hidden * weights).sum(dim=1)                    # (batch, hidden_dim)
        return pooled
 

class Evo2Regressor(nn.Module):
    """
    Evo2 backbone + regression head.

    The backbone is accessed via the Evo2 wrapper's internal .model attribute
    (a StripedHyena nn.Module), so standard PyTorch grad operations apply.

    Pooling: mean over real (non-padded) tokens of the chosen hidden layer.
    """

    def __init__(
        self,
        evo2_wrapper: Evo2,
        model_name: str,
        mlp_dims=(512, 256),
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_norm: bool = False,
        layer_name: str = None,
        pooling: str = "mean",
    ):
        super().__init__()
        self.backbone       = evo2_wrapper.model   # StripedHyena nn.Module
        self.tokenizer      = evo2_wrapper.tokenizer
        self.layer_name     = layer_name or EMBEDDING_LAYER[model_name]
        self.pooling        = pooling
        hidden_dim          = HIDDEN_DIM[model_name]

        # Attention pooling needs its own learned query vector, sized to
        # match the embedding dimension of the chosen layer
        self.attn_pool = AttentionPool(hidden_dim) if pooling == "attention" else None


        self.head           = RegressionHead(
                                hidden_dim, mlp_dims, dropout,
                                activation=activation, batch_norm=batch_norm,
                              )
        self._embedding_buf = {}

    def _hook(self, name):
        def fn(_, __, output):
            self._embedding_buf[name] = output[0] if isinstance(output, tuple) else output
        return fn

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        """
        input_ids  : (batch, seq_len)  long
        attn_mask  : (batch, seq_len)  bool  — True where real tokens exist
        Returns    : (batch,)          float predictions
        """
        self._embedding_buf.clear()
        hook_handle = self.backbone.get_submodule(self.layer_name).register_forward_hook(
            self._hook(self.layer_name)
        )
        try:
            _ = self.backbone(input_ids)           # forward through StripedHyena
        finally:
            hook_handle.remove()

        hidden = self._embedding_buf[self.layer_name]   # (batch, seq_len, hidden)

        # Pool over real (non-padded) tokens according to self.pooling
        mask = attn_mask.float()                          # (batch, seq_len)

        
        if self.pooling == "mean":
            # Average over real tokens — robust for variable-length sequences
            m = mask.unsqueeze(-1)                        # (batch, seq_len, 1)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "last":
            # Last real token — GPT-style; good when the model is causal
            lengths = mask.sum(dim=1).long() - 1          # (batch,)
            lengths = lengths.clamp(min=0)
            pooled  = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        elif self.pooling == "first":
            # First token — BERT [CLS]-style
            pooled = hidden[:, 0, :]
        elif self.pooling == "max":
            # Max over real tokens
            m      = mask.unsqueeze(-1)                   # (batch, seq_len, 1)
            masked = hidden * m + (1 - m) * (-1e9)        # mask out padding
            pooled = masked.max(dim=1).values
        elif self.pooling == "attention":
            hidden = hidden.float()  
            mask = attn_mask.float()
            # Learned attention pooling — see AttentionPool docstring
            pooled = self.attn_pool(hidden, mask)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling!r}")

        return self.head(pooled)

# =============================================================================
# Strategy implementations
# =============================================================================

def apply_frozen_strategy(model: Evo2Regressor):
    """Strategy 1: freeze backbone, train head (and attention pool, if used) only."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    if model.attn_pool is not None:
        for param in model.attn_pool.parameters():
            param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Strategy: frozen]  Trainable params: {n_train:,}")


def apply_full_strategy(model: Evo2Regressor):
    """Strategy 2: train everything."""
    for param in model.parameters():
        param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Strategy: full]    Trainable params: {n_train:,}")


def _block_index_from_layer_name(layer_name: str) -> int:
    """
    Extract the block index from a layer name like 'blocks.7.mlp.l3'.
    Returns -1 for layers outside the blocks (e.g. embedding_layer, norm).
    """
    parts = layer_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except ValueError:
            pass
    return -1


def apply_lora_strategy(model: Evo2Regressor, lora_r: int = 8, lora_alpha: int = 16,
                        lora_dropout: float = 0.05, max_block: int = None):
    """
    Strategy 3: inject LoRA adapters into linear layers of the backbone.

    Args:
        max_block : if given, only inject LoRA into blocks 0..max_block (inclusive).
                    Layers beyond max_block are frozen with no adapters.
                    Defaults to None = inject into the whole backbone.
                    Tip: set this to the block index of your embedding layer so
                    you don't waste parameters on layers whose gradients don't
                    reach your pooling point.

    Requires:  pip install peft
    """
    if not HAS_PEFT:
        raise ImportError(
            "peft is not installed. Install with:  pip install peft"
        )

    # Freeze backbone first
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Find linear layer names to target, optionally restricted to blocks <= max_block
    target_modules = set()
    excluded_count = 0
    for name, module in model.backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        block_idx = _block_index_from_layer_name(name)
        if max_block is not None and block_idx > max_block:
            excluded_count += 1
            continue
        target_modules.add(name.split(".")[-1])   # peft matches by leaf name

    target_modules = list(target_modules)

    if max_block is not None:
        print(f"[Strategy: lora]    Injecting LoRA into blocks 0–{max_block} only "
              f"({excluded_count} linear layers in deeper blocks left frozen).")

    # peft's get_peft_model() expects model.config.to_dict() to exist (HF convention).
    # Evo2's StripedHyena backbone has no such config, so we attach a minimal stub.
    if not hasattr(model.backbone, "config") or not hasattr(model.backbone.config, "to_dict"):
        class _FakeConfig:
            model_type = "stripedhyena"
            def to_dict(self):
                return {"model_type": self.model_type}
        model.backbone.config = _FakeConfig()
        
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    
    model.backbone = get_peft_model(model.backbone, lora_cfg)

    for name, p in model.named_parameters():
        if p.requires_grad and "lora" in name:
            print(f"[Strategy: lora]    Training LoRA adapter: {name}")


    # Head always trainable
    for param in model.head.parameters():
        param.requires_grad = True
    if model.attn_pool is not None:
        for param in model.attn_pool.parameters():
            param.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Strategy: lora]    Trainable params: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.2f}%)")

# =============================================================================
# Training loop
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device,
                grad_clip=None, step_log: list = None, epoch: int = 0):
    """
    Run one training epoch.

    Args:
        step_log  : list to append per-step dicts to (for CSV logging). Pass None to skip.
        epoch     : current epoch number (used in step log rows).

    Returns:
        (mean_loss, all_preds_np, all_labels_np, mean_grad_norm)
    """
    model.train()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    grad_norms = []

    for step, (input_ids, attn_mask, labels) in enumerate(
            tqdm(loader, leave=False, ncols=80)):
        input_ids  = input_ids.to(device)
        attn_mask  = attn_mask.to(device)
        labels     = labels.to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attn_mask)
        loss  = criterion(preds, labels)
        loss.backward()

        # Gradient norm BEFORE clipping — useful for diagnosing explosions
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip if grad_clip else float("inf")
        ).item()
        grad_norms.append(grad_norm)
        optimizer.step()

        step_loss = loss.item()
        total_loss += step_loss * labels.shape[0]
        n          += labels.shape[0]
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if step % 55 == 0:
            #Print the first trainable weight to see if they are changing
            for name, p in model.named_parameters():
                if p.requires_grad:
                    print(f"  Step {step:04d}  Loss={step_loss:.4f}  GradNorm={grad_norm:.4f}  "
                          f"First trainable weight: {name} = {p.flatten()[0].item():.6f}" 
                          f" Pred mean = {preds.detach().mean().item():.4f}  Pred std = {preds.detach().std().item():.4f}  ")
                    break
            
        if step_log is not None and step % 33 == 0:
            step_log.append({
                "epoch":      epoch,
                "step":       step,
                "loss":       step_loss,
                "grad_norm":  grad_norm,
                "pred_mean":  float(preds.detach().mean().cpu()),
                "pred_std":   float(preds.detach().std().cpu()),
                "label_mean": float(labels.mean().cpu()),
            })
        
        #if step == 55: break

    preds_np  = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    return total_loss / n, preds_np, labels_np, mean_grad_norm


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate on a dataloader.
    Returns dict with loss/pcc/spearman plus raw preds and labels arrays.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0

    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels    = labels.to(device)

        preds = model(input_ids, attn_mask)
        loss  = criterion(preds, labels)

        total_loss += loss.item() * labels.shape[0]
        n          += labels.shape[0]
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds_np  = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    pcc, _    = pearsonr(preds_np, labels_np)
    scc, _    = spearmanr(preds_np, labels_np)

    return {
        "loss":     total_loss / n,
        "pcc":      pcc,
        "spearman": scc,
        "preds":    preds_np,
        "labels":   labels_np,
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Evo2 for RiboScan regression (3 strategies)"
    )

    # Data
    p.add_argument("--train_csv",    required=True)
    p.add_argument("--val_csv",      required=True)
    p.add_argument("--test_csv",     default=None)
    p.add_argument("--seq_col",      default="Sequence")
    p.add_argument("--label_col",    default="mean_GFP")
    p.add_argument("--output_dir",   required=True)

    # Model
    p.add_argument("--model_name",   default="evo2_7b_base",
                   choices=["evo2_7b_base", "evo2_7b", "evo2_1b_base"],
                   help="evo2_7b_base is recommended: no FP8/TE required.")

    p.add_argument("--strategy",     default="frozen",
                   choices=["frozen", "full", "lora"],
                   help="frozen=head only | full=all weights | lora=LoRA adapters")

    p.add_argument("--embedding_layer", default=None,
                   help="Layer name to extract embeddings from, e.g. "
                        "'blocks.20.mlp.l3'. Overrides the default for the "
                        "chosen model. To list all layer names run with "
                        "--list_layers and exit.")
    
    p.add_argument("--list_layers", action="store_true",
                   help="Print all named modules in the backbone and exit. "
                        "Use this to find a layer name for --embedding_layer.")
    
    p.add_argument("--pooling", default="mean",
                   choices=["mean", "last", "first", "max", "attention"],
                   help="How to pool token embeddings into a single vector. "
                        "mean=average over real tokens (default), "
                        "last=last real token (like GPT), "
                        "first=first token (like BERT [CLS]), "
                        "max=max over real tokens, "
                        "attention=learned attention pooling (a single trainable "
                        "query vector weights tokens by relevance; good when "
                        "sequences are variable-length/unanchored and you want "
                        "the model to learn which positions matter without "
                        "assuming a fixed positional meaning).")

    # ── Regression head architecture ──────────────────────────────────
    # Two ways to specify the hidden layers (--mlp_dims takes priority):
    #   A) Explicit:  --mlp_dims 512 128 32
    #      Gives exactly those layer widths in order.
    #   B) Uniform:   --head_layers 3 --head_hidden_dim 256
    #      Builds N layers all of width D, halving towards the output.
    #      e.g. layers=3, dim=256  →  [256, 128, 64]
    p.add_argument("--mlp_dims",         type=int, nargs="+", default=None,
                   help="Explicit hidden-layer widths for the MLP head "
                        "(overrides --head_layers / --head_hidden_dim). "
                        "E.g. --mlp_dims 512 128 32")

    p.add_argument("--head_layers",      type=int, default=2,
                   help="Number of hidden layers in the MLP head (default: 2). "
                        "Ignored if --mlp_dims is given.")

    p.add_argument("--head_hidden_dim",  type=int, default=256,
                   help="Width of the first hidden layer; subsequent layers halve "
                        "this value (default: 256). Ignored if --mlp_dims is given.")

    p.add_argument("--head_dropout",     type=float, default=0.1,
                   help="Dropout probability after each hidden activation (default: 0.1)")

    p.add_argument("--head_activation",  default="gelu",
                   choices=["gelu", "relu", "silu", "tanh", "leaky"],
                   help="Activation function for hidden layers (default: gelu)")

    p.add_argument("--head_batch_norm",  action="store_true",
                   help="Add BatchNorm1d before each activation in the head")

    

    # LoRA options (only used when --strategy lora)
    p.add_argument("--lora_r",       type=int,   default=8)

    p.add_argument("--lora_alpha",   type=int,   default=16)

    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--lora_max_block", type=int, default=None,
                   help="Only inject LoRA adapters into backbone blocks 0..N. "
                        "Blocks beyond N are frozen with no adapters. "
                        "Strongly recommended: set this to the block index of "
                        "your --embedding_layer (e.g. 1 for blocks.1.mlp.l3) "
                        "so you don't train layers whose output you never use. "
                        "Default: None = inject into entire backbone.")

    # Training
    p.add_argument("--epochs",       type=int,   default=20)

    p.add_argument("--batch_size",   type=int,   default=16)

    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Use ~1e-4 for frozen/lora, ~1e-5 for full fine-tuning")

    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--grad_clip",    type=float, default=1.0)

    p.add_argument("--num_workers",  type=int,   default=2)

    p.add_argument("--seed",         type=int,   default=42)

    p.add_argument("--criterion",    default="mse", choices=["mse", "huber"])
    
    p.add_argument("--normalize_labels", action="store_true",
                   help="Z-score normalise labels on train stats before training. "
                        "Predictions are inverse-transformed for plotting/metrics. "
                        "Strongly recommended when label values span a large range.")

    return p.parse_args()


def build_mlp_dims(args: argparse.Namespace) -> tuple:
    """
    Resolve the MLP hidden-layer widths from CLI args.

    Priority:
      1. --mlp_dims  (explicit list, used as-is)
      2. --head_layers + --head_hidden_dim  (generate a halving sequence)

    Examples
    --------
    --mlp_dims 512 128 32          → (512, 128, 32)
    --head_layers 3 --head_hidden_dim 256  → (256, 128, 64)
    --head_layers 1 --head_hidden_dim 128  → (128,)
    --head_layers 0                        → ()   linear probe
    """
    if args.mlp_dims is not None:
        return tuple(args.mlp_dims)

    if args.head_layers == 0:
        return ()

    dims = []
    d = args.head_hidden_dim
    for _ in range(args.head_layers):
        dims.append(max(d, 1))
        d = max(d // 2, 1)
    return tuple(dims)



def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Resolve output directory (creates trial_N subfolder if base exists)
    # ------------------------------------------------------------------
    output_dir = resolve_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if output_dir != args.output_dir:
        print(f"  Output dir already exists → using: {output_dir}")

    # Persist the resolved path back so save_args records it correctly
    args.output_dir = output_dir

    print(f"\n{'='*65}")
    print(f"  Evo2 RiboScan Regression  |  strategy={args.strategy}")
    print(f"  model={args.model_name}   device={device}")
    print(f"  output_dir={output_dir}")
    print(f"{'='*65}\n")

    # Save all arguments to args.json immediately
    save_args(args, output_dir)


    # ------------------------------------------------------------------
    # 1. Load Evo2
    # ------------------------------------------------------------------
    print("Loading Evo2 (may download checkpoint on first run)...")
    evo2_wrapper = Evo2(args.model_name)

    # ------------------------------------------------------------------
    # 1b. Optionally list all backbone layer names and exit
    # ------------------------------------------------------------------
    if args.list_layers:
        print("\nNamed modules in the Evo2 backbone:\n")
        for name, module in evo2_wrapper.model.named_modules():
            if name:   # skip the root ''
                print(f"  {name:70s}  {type(module).__name__}")
        print(f"\nDefault embedding layer for {args.model_name}: "
              f"{EMBEDDING_LAYER[args.model_name]}")
        return

    # ------------------------------------------------------------------
    # 2. Build the regressor
    # ------------------------------------------------------------------
    layer_name = args.embedding_layer or EMBEDDING_LAYER[args.model_name]

    mlp_dims = build_mlp_dims(args)

    print(f"  Embedding layer : {layer_name}")
    print(f"  Pooling         : {args.pooling}")
    print(f"  Head architecture: dims={mlp_dims}  activation={args.head_activation}  "
          f"batch_norm={args.head_batch_norm}  dropout={args.head_dropout}\n")
 

    model = Evo2Regressor(
        evo2_wrapper  = evo2_wrapper,
        model_name    = args.model_name,
        mlp_dims      = mlp_dims,
        dropout       = args.head_dropout,
        activation    = args.head_activation,
        batch_norm    = args.head_batch_norm,
        layer_name    = layer_name,
        pooling       = args.pooling,
    )

    

    # Apply chosen strategy
    if args.strategy == "frozen":
        apply_frozen_strategy(model)
    elif args.strategy == "full":
        apply_full_strategy(model)
    elif args.strategy == "lora":
        apply_lora_strategy(model, args.lora_r, args.lora_alpha, args.lora_dropout,
                            max_block=args.lora_max_block)

    model = model.to(device)

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  TRAINABLE  {name:60s}  {tuple(p.shape)}")
    
    

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    pad_id = evo2_wrapper.tokenizer.pad_id

    train_seqs, train_labels = load_csv(args.train_csv, args.seq_col, args.label_col)
    print(f'Train CSV: {args.train_csv}  →  {len(train_seqs):,} sequences \n {train_seqs[:5]} \n {train_labels[:5]}')
    val_seqs,   val_labels   = load_csv(args.val_csv,   args.seq_col, args.label_col)

    # Optional label normalisation — highly recommended when labels span a wide range
    scaler = LabelScaler()
    if args.normalize_labels:
        scaler.fit(train_labels)
        train_labels = scaler.transform(train_labels)
        val_labels   = scaler.transform(val_labels)
        scaler.save(os.path.join(output_dir, "label_scaler.json"))
        print("  Label normalisation: ON (scaler saved to label_scaler.json)")
    else:
        label_arr = np.array(train_labels)
        print(f"  Labels (raw): min={label_arr.min():.3f}  max={label_arr.max():.3f}  "
              f"mean={label_arr.mean():.3f}  std={label_arr.std():.3f}")
        print("  Tip: if std is large (>10× the mean), try --normalize_labels")

    train_ds = Evo2Dataset(train_seqs, train_labels, evo2_wrapper.tokenizer)
    print(f"  Train dataset: {len(train_ds):,} sequences {type(train_ds[0][0])} tokens, {type(train_ds[0][1])} labels")
    val_ds   = Evo2Dataset(val_seqs,   val_labels,   evo2_wrapper.tokenizer)

    _collate = lambda b: collate_fn(b, pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, collate_fn=_collate)

    print(f"  Train: {len(train_ds):,}   Val: {len(val_ds):,}\n")

    # ------------------------------------------------------------------
    # 4. Optimiser + loss
    # ------------------------------------------------------------------
    # Separate LR for backbone (lower) vs head (higher) when full fine-tuning
    if args.strategy == "full":
        head_params = list(model.head.parameters())
        if model.attn_pool is not None:
            head_params += list(model.attn_pool.parameters())
        param_groups = [
            {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
            {"params": head_params,                 "lr": args.lr},
        ]
    else:
        param_groups = [{"params": filter(lambda p: p.requires_grad, model.parameters()),
                         "lr": args.lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss() if args.criterion == "mse" else nn.HuberLoss()

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_pcc, best_epoch = -1.0, 0
    history   = []
    step_log  = []          # per-step loss/grad_norm records
    ckpt_path = os.path.join(output_dir, "best_model.pth")
    step_log_path = os.path.join(output_dir, "step_log.csv")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        train_loss, train_preds, train_labels_np, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=args.grad_clip, step_log=step_log, epoch=epoch,
        )

        # Compute train metrics for logging and plotting
        train_pcc, _  = pearsonr(train_preds, train_labels_np)
        train_scc, _  = spearmanr(train_preds, train_labels_np)

        # ── Validate ──
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Inverse-transform predictions for scatter plots when normalised
        # (so axes show original label units)
        plot_train_preds  = scaler.inverse_transform(train_preds)
        plot_train_labels = scaler.inverse_transform(train_labels_np)
        plot_val_preds    = scaler.inverse_transform(val_metrics["preds"])
        plot_val_labels   = scaler.inverse_transform(val_metrics["labels"])

        # ── Scatter plots (train + val) ──
        save_scatter_plot(
            plot_train_preds, plot_train_labels,
            epoch=epoch, split="train",
            output_dir=output_dir,
            pcc=train_pcc, spearman=train_scc,
        )
        save_scatter_plot(
            plot_val_preds, plot_val_labels,
            epoch=epoch, split="val",
            output_dir=output_dir,
            pcc=val_metrics["pcc"], spearman=val_metrics["spearman"],
        )

        # ── Logging ──
        row = {
            "epoch":          epoch,
            "train_loss":     train_loss,
            "train_pcc":      train_pcc,
            "train_spearman": train_scc,
            "val_loss":       val_metrics["loss"],
            "val_pcc":        val_metrics["pcc"],
            "val_spearman":   val_metrics["spearman"],
            "grad_norm":      grad_norm,
            "pred_std":       float(train_preds.std()),
        }
        history.append(row)

        # Flush step log to CSV after every epoch (safe to inspect mid-run)
        pd.DataFrame(step_log).to_csv(step_log_path, index=False)

        # Update loss-curve plot
        save_loss_curve(history, output_dir)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_PCC={train_pcc:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_PCC={val_metrics['pcc']:.4f}  "
              f"val_Spearman={val_metrics['spearman']:.4f}  "
              f"grad_norm={grad_norm:.3f}  pred_std={train_preds.std():.4f}  "
              f"({time.time()-t0:.0f}s)")

        # ── Checkpoint ──
        if val_metrics["pcc"] > best_pcc:
            best_pcc   = val_metrics["pcc"]
            best_epoch = epoch
            torch.save({
                "epoch":          epoch,
                "strategy":       args.strategy,
                "model_name":     args.model_name,
                "head_state":     model.head.state_dict(),
                "attn_pool_state": (
                    model.attn_pool.state_dict() if model.attn_pool is not None else None
                ),
                "backbone_state": (
                    model.backbone.state_dict()
                    if args.strategy in ("full", "lora") else None
                ),
                "val_pcc":        best_pcc,
            }, ckpt_path)
            print(f"  ✓ Best model saved (val PCC={best_pcc:.4f})")

    print(f"\nBest validation PCC: {best_pcc:.4f} at epoch {best_epoch}")

    # ------------------------------------------------------------------
    # 6. Test evaluation
    # ------------------------------------------------------------------
    if args.test_csv:
        test_seqs, test_labels = load_csv(args.test_csv, args.seq_col, args.label_col)
        test_ds     = Evo2Dataset(test_seqs, test_labels, evo2_wrapper.tokenizer)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=_collate)

        # Reload best checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)

        model.head.load_state_dict(ckpt["head_state"])
        if ckpt.get("attn_pool_state") is not None and model.attn_pool is not None:
            model.attn_pool.load_state_dict(ckpt["attn_pool_state"])
        if ckpt["backbone_state"] is not None:
            model.backbone.load_state_dict(ckpt["backbone_state"])

        test_metrics = evaluate(model, test_loader, criterion, device)

        # Test scatter plot
        save_scatter_plot(
            test_metrics["preds"], test_metrics["labels"],
            epoch=best_epoch, split="test",
            output_dir=output_dir,
            pcc=test_metrics["pcc"], spearman=test_metrics["spearman"],
        )

        # Drop raw arrays before serialising to JSON
        test_metrics_json = {k: v for k, v in test_metrics.items()
                             if k not in ("preds", "labels")}
        print(f"\nTest metrics: {test_metrics_json}")
        with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics_json, f, indent=2)

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"), index=False
    )
    print(f"\nOutputs saved to: {output_dir}")


# =============================================================================
# RiboScanner integration helpers
# =============================================================================

def load_evo2_regressor_from_checkpoint(
    ckpt_path: str,
    device: str = "cuda",
    layer_name: str = None,
    pooling: str = "mean",
) -> Evo2Regressor:
    """
    Load a saved Evo2Regressor for inference inside RiboScanner's predict pipeline.

    Example usage in predict_model.py:
        from .evo2_regression import load_evo2_regressor_from_checkpoint, predict_from_seq_evo2
        model = load_evo2_regressor_from_checkpoint("outputs/evo2-frozen/best_model.pth")
        preds = predict_from_seq_evo2(model, sequences)
    """
    ckpt        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name  = ckpt["model_name"]
    strategy    = ckpt["strategy"]

    evo2_wrapper = Evo2(model_name)
    model        = Evo2Regressor(
        evo2_wrapper, model_name,
        layer_name=layer_name, pooling=pooling,
    )

    model.head.load_state_dict(ckpt["head_state"])
    if ckpt.get("attn_pool_state") is not None and model.attn_pool is not None:
        model.attn_pool.load_state_dict(ckpt["attn_pool_state"])
    if strategy in ("full", "lora") and ckpt["backbone_state"] is not None:
        model.backbone.load_state_dict(ckpt["backbone_state"])

    model = model.to(device)
    model.eval()
    return model, evo2_wrapper.tokenizer


@torch.no_grad()
def predict_from_seq_evo2(
    model: Evo2Regressor,
    sequences: list,
    tokenizer,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Drop-in replacement for RiboScanner's predict_from_seq, using the Evo2 backbone.
    Returns a (N,) numpy array of predictions.
    """
    pad_id  = tokenizer.pad_id
    ds      = Evo2Dataset(sequences, [0.0] * len(sequences), tokenizer)
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         collate_fn=lambda b: collate_fn(b, pad_id))
    model.eval()
    preds = []
    for input_ids, attn_mask, _ in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        preds.append(model(input_ids, attn_mask).cpu().numpy())
    return np.concatenate(preds)


if __name__ == "__main__":
    main()



"""
python evo2_regression.py \
    --strategy frozen \
    --model_name evo2_1b_base \
    --train_csv /hpc/compgen/projects/translation_modeling/DL_TSS/raw/data_bram/LB20250527_BV20240725_data_for_AI_updated.xlsx_train_all_folds_0-8.csv \
    --val_csv /hpc/compgen/projects/translation_modeling/DL_TSS/raw/data_bram/LB20250527_BV20240725_data_for_AI_updated.xlsx_train_fold9.csv \
    --seq_col Sequence --label_col mean_GFP_nolog2 \
    --epochs 20 --lr 1e-3 \
    --output_dir /hpc/compgen/projects/translation_modeling/DL_TSS/analysis/lbarbadillamartinez/output/Bram_data/evo2/evo2-frozen/evo2_1b_base/

# Step 2: if PCC looks good, try LoRA
python evo2_regression.py \
    --strategy lora --lora_r 16 --lora_alpha 32 \
    --lr 5e-4 --epochs 30 \
    --train_csv /hpc/compgen/projects/translation_modeling/DL_TSS/raw/data_bram/LB20250527_BV20240725_data_for_AI_updated.xlsx_train_all_folds_0-8.csv \
    --val_csv /hpc/compgen/projects/translation_modeling/DL_TSS/raw/data_bram/LB20250527_BV20240725_data_for_AI_updated.xlsx_train_fold9.csv \
    --seq_col Sequence --label_col mean_GFP_nolog2 \
    --output_dir /hpc/compgen/projects/translation_modeling/DL_TSS/analysis/lbarbadillamartinez/output/Bram_data/evo2/evo2-lora



# SE — local motifs (Kozak, AUG context, short secondary structure)
--embedding_layer blocks.0.mlp.l3    # earliest, most local
--embedding_layer blocks.7.mlp.l3    # SE after one full SE+MR+LI+Attn cycle
--embedding_layer blocks.14.mlp.l3   # mid-network SE

# MR — medium range, ~128 nt kernel = covers most of your sequence
--embedding_layer blocks.1.mlp.l3    # earliest MR
--embedding_layer blocks.8.mlp.l3    # MR after first attention

# Raw filter output (before MLP) — purest convolutional signal
--embedding_layer blocks.0.out_filter_dense   # raw SE output
--embedding_layer blocks.1.out_filter_dense   # raw MR output

# LI — probably least useful for 150 bp, designed for genome-scale
--embedding_layer blocks.2.mlp.l3    # skip this unless others fail

"""