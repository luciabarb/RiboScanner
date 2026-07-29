"""
utils_riboscanner_extensions.py — External model adapters for RiboScanner
==========================================================================
Drop this file into RiboScanner/ alongside the existing source files.

Provides self-contained inference adapters for two external models that
plug into RiboScanner's predict pipeline via the --model_type argument:

  model_type='noderer'
      Dinucleotide PWM baseline (Noderer et al. 2014, Mol Syst Biol).
      A simple linear regression on position-pair one-hot features, fit
      in log-space. No neural network, no GPU required.
      Requires:  the .npz coefficient file saved by noderer_pwm.py
                 (e.g. outputs/noderer-pwm/di_pwm_coefficients.npz)
      Needs to know where the AUG is in each sequence, via aug_col.

  model_type='evo2'
      Evo2 genomic language model with a fine-tuned regression head,
      trained via evo2_regression.py.
      Requires:  the best_model.pth checkpoint saved by evo2_regression.py
      Needs:     evo2 installed in the environment

USAGE (from predict_model.py / cli.py after integration)
---------------------------------------------------------
  # Noderer PWM
  RiboScanner predict \
      --model_type noderer \
      --model /hpc/compgen/projects/translation_modeling/DL_TSS/analysis/lbarbadillamartinez/output/Bram_data/Noderer/trial_23/di_pwm_coefficients.npz \
      --noderer_order di \
      --input /hpc/compgen/projects/translation_modeling/DL_TSS/raw/data_bram/LB20250527_BV20240725_data_for_AI_updated.xlsx_test_with_TIS.csv \
      --column_sequence Sequence \
      --measurement_column mean_GFP_nolog2 \
      --split_on_variable TIS_ \
      --output /hpc/compgen/projects/translation_modeling/DL_TSS/analysis/lbarbadillamartinez/output/Bram_data/Noderer/trial_23/output_figures/di/test_predictions.tsv

  # Evo2
  RiboScanner predict \
      --model_type evo2 \
      --model path/to/best_model.pth \
      --input data.csv --column_sequence Sequence \
      --output predictions.tsv
"""

from __future__ import annotations

import importlib
import numpy as np


# =============================================================================
# ── Noderer PWM adapter ───────────────────────────────────────────────────────
# =============================================================================

# Noderer's 8 variable positions (paper default). The AUG itself (+1,+2,+3)
# is never modeled — it's invariant across the library.
_DEFAULT_POSITIONS = [-6, -5, -4, -3, -2, -1, 4, 5]
_BASES = ["U", "C", "A", "G"]


def _position_to_seq_index(pos: int, aug_index: int) -> int:
    """Convert a biological position relative to AUG into a 0-based seq index."""
    if pos < 0:
        return aug_index + pos          # e.g. -1 → aug_index - 1
    else:                               # pos >= 4 (AUG is +1,+2,+3)
        return aug_index + (pos - 1)    # e.g. +4 → aug_index + 3


def _extract_tis_kmer(full_seq: str, aug_index: int, positions: list) -> str | None:
    """
    Extract the variable TIS positions from a full sequence.
    Returns None if the sequence is too short or contains non-ACGU characters.
    """
    full_seq = full_seq.upper().replace("T", "U")
    bases = []
    for pos in positions:
        idx = _position_to_seq_index(pos, aug_index)
        if idx < 0 or idx >= len(full_seq):
            return None
        bases.append(full_seq[idx])
    kmer = "".join(bases)
    if not set(kmer) <= set(_BASES):
        return None
    return kmer


def _find_first_aug(seq: str) -> int | None:
    """Find the 0-based index of the first AUG in a sequence. Returns None if absent."""
    seq = seq.upper().replace("U", "T")
    idx = seq.find("ATG")
    return idx if idx != -1 else None


def _encode_mono(kmers: list, positions: list, bases: list) -> np.ndarray:
    n_pos = len(positions)
    n = len(kmers)
    X = np.zeros((n, n_pos * len(bases)), dtype=np.float64)
    base_idx = {b: i for i, b in enumerate(bases)}
    for row, seq in enumerate(kmers):
        for pos_i, base in enumerate(seq):
            col = pos_i * len(bases) + base_idx.get(base, 0)
            X[row, col] = 1.0
    return X


def _encode_di(kmers: list, positions: list, bases: list) -> np.ndarray:
    n_pos = len(positions)
    n_pairs = n_pos * (n_pos - 1) // 2
    n = len(kmers)
    X = np.zeros((n, n_pairs * len(bases) * len(bases)), dtype=np.float64)
    base_idx = {b: i for i, b in enumerate(bases)}
    pair_offset = {}
    k = 0
    for i1 in range(n_pos):
        for i2 in range(i1 + 1, n_pos):
            pair_offset[(i1, i2)] = k
            k += 1
    for row, seq in enumerate(kmers):
        for i1 in range(n_pos):
            for i2 in range(i1 + 1, n_pos):
                b1 = base_idx.get(seq[i1], 0)
                b2 = base_idx.get(seq[i2], 0)
                block = pair_offset[(i1, i2)]
                col = block * len(bases) ** 2 + b1 * len(bases) + b2
                X[row, col] = 1.0
    return X


class NodererPredictor:
    """
    Loaded Noderer PWM ready for inference inside RiboScanner's predict pipeline.

    Args:
        coef_path  : path to the .npz file saved by noderer_pwm.py
                     (e.g. di_pwm_coefficients.npz or mono_pwm_coefficients.npz)
        order      : 'mono' or 'di' — must match the coefficients file
        positions  : which TIS positions were used during training.
                     Defaults to the paper's original 8: [-6,-5,-4,-3,-2,-1,4,5].
        aug_col    : when sequences are full-length (not pre-cropped), this is the
                     name of a column in the dataframe giving the 0-based index of
                     the 'A' in AUG. If None, the first ATG in each sequence is
                     used automatically.
        label_scaler_path : optional path to label_scaler.json saved by
                            noderer_pwm.py when --normalize_labels was used.
                            If provided, predictions are inverse-transformed back
                            to original label units automatically.
    """

    def __init__(
        self,
        coef_path: str,
        order: str = "di",
        positions: list | None = None,
        aug_col: str | None = None,
        label_scaler_path: str | None = None,
    ):
        import json
        data = np.load(coef_path)
        self.coef_      = data["coef"]
        self.intercept_ = float(data["intercept"])
        self.order      = order
        self.positions  = sorted(positions or _DEFAULT_POSITIONS)
        self.aug_col    = aug_col

        self.scaler_mean = 0.0
        self.scaler_std  = 1.0
        if label_scaler_path is not None:
            with open(label_scaler_path) as f:
                s = json.load(f)
            self.scaler_mean = float(s["mean"])
            self.scaler_std  = float(s["std"])

    def _get_aug_index(self, seq: str, row_aug_index=None) -> int | None:
        if row_aug_index is not None:
            try:
                return int(row_aug_index)
            except (TypeError, ValueError):
                pass
        return _find_first_aug(seq)

    def predict(self, sequences: list, aug_indices: list | None = None) -> np.ndarray:
        """
        Predict efficiency for a list of sequences.

        sequences   : list of full-length DNA/RNA strings
        aug_indices : optional list of 0-based AUG positions (same length as sequences).
                      If None and aug_col was not set, first ATG is used.

        Returns a (N,) numpy array in original label units.
        """
        kmers = []
        valid_mask = []
        for i, seq in enumerate(sequences):
            aug_idx = self._get_aug_index(
                seq, aug_indices[i] if aug_indices is not None else None
            )
            if aug_idx is None:
                kmer = None
            else:
                kmer = _extract_tis_kmer(seq, aug_idx, self.positions)
            if kmer is None:
                kmers.append("U" * len(self.positions))  # placeholder
                valid_mask.append(False)
            else:
                kmers.append(kmer)
                valid_mask.append(True)

        # Build feature matrix
        X_mono = _encode_mono(kmers, self.positions, _BASES)
        if self.order == "di":
            X_di = _encode_di(kmers, self.positions, _BASES)
            X = np.concatenate([X_mono, X_di], axis=1)
        else:
            X = X_mono

        log_preds = X @ self.coef_ + self.intercept_
        preds = np.exp(log_preds)  # back to raw-scale efficiency

        # Inverse-transform if the PWM was fit on normalised labels
        preds = preds * self.scaler_std + self.scaler_mean

        # Sequences where we couldn't extract a k-mer get NaN
        preds[~np.array(valid_mask)] = float("nan")

        n_invalid = (~np.array(valid_mask)).sum()
        if n_invalid > 0:
            print(
                f"  [NodererPredictor] WARNING: {n_invalid} sequence(s) had no "
                f"valid AUG or the AUG was too close to the sequence edge — "
                f"their predictions are NaN.",
                flush=True,
            )

        return preds.astype(np.float32)


def load_noderer_predictor(
    coef_path: str,
    order: str = "di",
    positions: list | None = None,
    aug_col: str | None = None,
    label_scaler_path: str | None = None,
) -> NodererPredictor:
    """Convenience wrapper — load a NodererPredictor from a saved .npz file."""
    return NodererPredictor(
        coef_path=coef_path,
        order=order,
        positions=positions,
        aug_col=aug_col,
        label_scaler_path=label_scaler_path,
    )


# =============================================================================
# ── Evo2 adapter ──────────────────────────────────────────────────────────────
# =============================================================================

class Evo2Predictor:
    """
    Loaded Evo2Regressor checkpoint ready for inference inside RiboScanner.

    Wraps the load_evo2_regressor_from_checkpoint and predict_from_seq_evo2
    helpers from evo2_regression.py, providing the same simple .predict()
    interface as NodererPredictor so predict_model.py can call both the
    same way.

    Args:
        ckpt_path          : path to best_model.pth saved by evo2_regression.py
        device             : 'cuda' or 'cpu'
        evo2_regression_path : path to evo2_regression.py if it is NOT already
                               importable as a module. When None, we try to import
                               it from the RiboScanner package first, then fall
                               back to looking for it as a standalone script in the
                               same directory as this file.
        batch_size         : sequences per forward pass (reduce if OOM)
        label_scaler_path  : optional path to label_scaler.json; if provided,
                             predictions are inverse-transformed back to original
                             label units automatically.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        evo2_regression_path: str | None = None,
        batch_size: int = 32,
        label_scaler_path: str | None = None,
        layer_name: str | None = None,
        pooling: str = "mean",
    ):
        import json, importlib.util, sys, os

        self.device     = device
        self.batch_size = batch_size

        # ── Import evo2_regression from wherever it lives ──────────────────
        try:
            # Case 1: installed as part of a package (e.g. RiboScanner.evo2_regression)
            evo2_reg = importlib.import_module("RiboScanner.evo2_regression")
        except ModuleNotFoundError:
            try:
                # Case 2: importable directly (e.g. on sys.path already)
                evo2_reg = importlib.import_module("evo2_regression")
            except ModuleNotFoundError:
                # Case 3: standalone script — load from explicit path
                if evo2_regression_path is None:
                    # Default: look next to this file
                    evo2_regression_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "evo2_regression.py",
                    )
                spec = importlib.util.spec_from_file_location(
                    "evo2_regression", evo2_regression_path
                )
                evo2_reg = importlib.util.module_from_spec(spec)
                sys.modules["evo2_regression"] = evo2_reg
                spec.loader.exec_module(evo2_reg)

        self._evo2_reg = evo2_reg

        # ── Load checkpoint ───────────────────────────────────────────────
        print(f"  [Evo2Predictor] Loading checkpoint: {ckpt_path}", flush=True)
        self.model, self.tokenizer = evo2_reg.load_evo2_regressor_from_checkpoint(
            ckpt_path=ckpt_path,
            device=device,
            layer_name=layer_name,
            pooling=pooling
        )
        self.model.eval()
        print(f"  [Evo2Predictor] Checkpoint loaded.", flush=True)

        # ── Optional label scaler ─────────────────────────────────────────
        self.scaler_mean = 0.0
        self.scaler_std  = 1.0
        if label_scaler_path is not None:
            with open(label_scaler_path) as f:
                s = json.load(f)
            self.scaler_mean = float(s["mean"])
            self.scaler_std  = float(s["std"])
            print(
                f"  [Evo2Predictor] Label scaler loaded: "
                f"mean={self.scaler_mean:.4f}  std={self.scaler_std:.4f}",
                flush=True,
            )

    def predict(self, sequences: list, **kwargs) -> np.ndarray:
        """
        Run inference on a list of sequences.
        Returns a (N,) numpy array in original label units.
        """
        preds = self._evo2_reg.predict_from_seq_evo2(
            model=self.model,
            sequences=sequences,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            device=self.device,
        )
        # Inverse-transform if labels were z-scored during training
        preds = preds * self.scaler_std + self.scaler_mean
        return preds.astype(np.float32)


def load_evo2_predictor(
    ckpt_path: str,
    device: str = "cuda",
    evo2_regression_path: str | None = None,
    batch_size: int = 32,
    label_scaler_path: str | None = None,
    layer_name: str = 'blocks.1.mlp.l3',
    pooling: str = "mean"
) -> Evo2Predictor:
    """Convenience wrapper — load an Evo2Predictor from a saved checkpoint."""
    return Evo2Predictor(
        ckpt_path=ckpt_path,
        device=device,
        evo2_regression_path=evo2_regression_path,
        batch_size=batch_size,
        label_scaler_path=label_scaler_path,
        layer_name=layer_name,
        pooling=pooling
    )
