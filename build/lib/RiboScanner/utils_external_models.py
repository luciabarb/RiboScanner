import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# ---------------------------------------------------------------------------
# Core transformer building blocks (unchanged from original)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Attention(nn.Module):
    def __init__(self, n_head, n_embd, dropout, bias, block_size):
        super().__init__()
        self.n_heads = n_head
        self.embed_dim = n_embd
        self.dropout_prob = dropout
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=bias)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.attention_dropout = nn.Dropout(self.dropout_prob)
        self.residual_dropout = nn.Dropout(self.dropout_prob)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            tril = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("causal_mask", tril.view(1, 1, block_size, block_size))
    def forward(self, hidden_states):
        batch_sz, seq_len, emb_dim = hidden_states.size()
        q, k, v = self.c_attn(hidden_states).chunk(3, dim=-1)
        head_dim = emb_dim // self.n_heads
        def shape(t):
            return t.view(batch_sz, seq_len, self.n_heads, head_dim).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        else:
            scale = 1.0 / math.sqrt(head_dim)
            sim = (q @ k.transpose(-2, -1)) * scale
            sim = sim.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
            attn = self.attention_dropout(F.softmax(sim, dim=-1))
            y = attn @ v
        y = y.transpose(1, 2).contiguous().reshape(batch_sz, seq_len, emb_dim)
        return self.residual_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = Attention(n_head, n_embd, dropout, bias, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp  = MLP(n_embd, dropout, bias)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class UTR_(nn.Module):
    """
    Original GemoRNA UTR language-model backbone.
    Weights are loaded from the pretrained .pt checkpoints as-is.
    """
    def __init__(self, vocab_size, n_embd, n_head, dropout, bias, block_size, n_layer):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte  = nn.Embedding(vocab_size, n_embd)
        self.transformer.wpe  = nn.Embedding(block_size, n_embd)
        self.transformer.drop = nn.Dropout(dropout)
        self.transformer.h    = nn.ModuleList([
            DecoderBlock(n_embd, n_head, dropout, bias, block_size)
            for _ in range(n_layer)
        ])
        self.transformer.ln_f = LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight   # weight tying
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, targets=None):
        device = input_ids.device
        b, t = input_ids.shape
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(torch.arange(t, device=device, dtype=torch.long))
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])
        return logits, None


# ---------------------------------------------------------------------------
# adapted_GemoRNA — regression head on top of the frozen backbone
# ---------------------------------------------------------------------------

class adapted_GemoRNA(nn.Module):
    """
    GemoRNA backbone with the LM head replaced by a regression head.

    Parameters
    ----------
    vocab_size : int
        Must match the checkpoint. For 5'UTR use max(five_prime_utr_vocab.values()) + 1 = 473.
        For 3'UTR use max(three_prime_utr_vocab.values()) + 1 = 432.
    n_embd, n_head, dropout, bias, block_size, n_layer : backbone hyperparams
        Must match the pretrained config exactly.
    num_classes : int
        Output dimension (1 for scalar regression, >1 for multi-output).
    pooling : str
        'last'  — last-token representation (default; matches the causal model)
        'mean'  — mean-pool over all positions
        'first' — first-token (<sos>) representation
    freeze_backbone : bool
        If True, all backbone parameters are frozen; only the regression head trains.
    """
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        block_size: int,
        n_layer: int,
        num_classes: int = 1,
        pooling: str = 'last',
        freeze_backbone: bool = False,
    ):
        super().__init__()
        assert pooling in ('last', 'mean', 'first'), \
            "pooling must be 'last', 'mean', or 'first'"
        self.pooling = pooling
        # Build the backbone (lm_head is never called; we bypass it in forward)
        self.backbone = UTR_(vocab_size, n_embd, n_head, dropout, bias, block_size, n_layer)
        # Regression / classification head
        self.head = nn.Linear(n_embd, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Re-enable the head
            for p in self.head.parameters():
                p.requires_grad = True
    def _encode(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return the full hidden-state tensor [B, T, C]."""
        b, t = idx.size()
        device = idx.device
        tok_emb = self.backbone.transformer.wte(idx)
        pos_emb = self.backbone.transformer.wpe(torch.arange(t, device=device))
        x = self.backbone.transformer.drop(tok_emb + pos_emb)
        for block in self.backbone.transformer.h:
            x = block(x)
        x = self.backbone.transformer.ln_f(x)   # [B, T, C] #The T is the same as the block_size used in the checkpoint, which is 768 for 5'UTR and 512 for 3'UTR
        return x
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        idx : LongTensor of shape [B, T]   (token indices)
        Returns
        -------
        logits : FloatTensor of shape [B, num_classes]
        """
        
        x = self._encode(idx)                    # [B, T, C]
        if self.pooling == 'last':
            pooled = x[:, -1, :]                 # last token
        elif self.pooling == 'mean':
            pooled = x.mean(dim=1)               # mean over sequence
        else:                                    # 'first'
            pooled = x[:, 0, :]                  # <sos> token
        return self.head(pooled)                 # [B, num_classes]
    def load_pretrained_backbone(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Load a GemoRNA pretrained checkpoint into the backbone.
        The checkpoint is expected to be a state_dict saved from the original
        UTR_ model (keys like 'transformer.wte.weight', 'lm_head.weight', …).
        Extra keys (lm_head.*) are silently ignored because we don't use them.
        """
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Some checkpoints wrap the state_dict under a 'model' key
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        print(f'State dict keys: {list(state.keys())}')
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        print(f'Missing keys: {missing} | Unexpected keys: {unexpected}')
        # lm_head keys are expected to be "unexpected" since we keep ours separate
        real_missing = [k for k in missing if not k.startswith('lm_head')]
        if real_missing:
            print(f"[WARNING] Missing backbone keys: {real_missing}")
        if unexpected:
            print(f"[INFO] Ignored checkpoint keys: {unexpected}")
        print(f"[OK] Loaded pretrained backbone from {checkpoint_path}")
        return self


# ---------------------------------------------------------------------------
# Vocabulary & tokenisation (5'UTR)
# ---------------------------------------------------------------------------

five_prime_utr_vocab = {'<sos>': 1, 'ACA': 2, 'CAG': 3, 'CGG': 4, 'GAA': 5, 'GGG': 6, 'AUU': 7, 'GCG': 8, 'AGC': 9, 'UGG': 10, 'UCG': 11, 'GAC': 12, 'CAA': 13, 'CCU': 14, 'GAG': 15, 'AAG': 16, 'ACC': 17, 'CCC': 18, 'GUC': 19, '<eos>': 20, 'AAC': 21, 'AAA': 22, 'CUG': 23, 'AUA': 24, 'CCA': 25, 'GCA': 26, 'UCU': 27, 'UGU': 28, 'UUG': 29, 'UUU': 30, 'UCC': 31, 'UCA': 32, 'AAU': 33, 'CAU': 34, 'CAC': 35, 'CGU': 36, 'UAU': 37, 'GAU': 38, 'GUG': 39, 'AGU': 40, 'CGA': 41, 'GGC': 42, 'GGU': 43, 'GGA': 44, 'AGG': 45, 'AGA': 46, 'ACU': 47, 'UAC': 48, 'UUC': 49, 'AUC': 50, 'CGC': 51, 'GCU': 52, 'CCG': 53, 'UUA': 54, 'GUA': 55, 'CUA': 56, 'GCC': 57, 'CUC': 58, 'CUU': 59, 'GUU': 60, 'AUG': 61, 'UAG': 62, 'UGC': 63, 'UGA': 64, 'ACG': 65, 'UAA': 66, 'GGN': 67, 'UNN': 68, 'CNN': 69, 'CUN': 70, 'ANN': 71, 'GAN': 72, 'AAN': 73, 'GNN': 74, 'AGN': 75, 'UUN': 76, 'UGN': 77, 'CAN': 78, 'GCN': 79, 'CGN': 80, 'UCN': 81, 'CCN': 82, 'UAN': 83, 'AUN': 84, 'GUN': 85, 'ACN': 86, 'NNC': 87, 'NCC': 88, 'NGG': 89, 'NGC': 90, 'GNG': 91, 'CNG': 92, 'CNA': 93, 'NCA': 94, 'NCU': 95, 'NAG': 96, 'NNN': 97, 'NNU': 98, 'NUU': 99, 'NGN': 100, 'NGA': 101, 'NNA': 102, 'NAA': 103, 'NNG': 104, 'NCG': 105, 'GNC': 106, 'ANC': 107, 'NUC': 108, 'CNU': 109, 'GNA': 110, 'NAC': 111, 'ANU': 112, 'CNC': 113, 'GNU': 114, 'NGU': 115, 'UNC': 116, 'UNU': 117, 'NAU': 118, 'NUG': 119, 'ANA': 120, 'UNG': 121, 'ANG': 122, 'NUA': 123, 'CRG': 124, 'SCA': 125, 'AMU': 126, 'KCY': 127, 'MGC': 128, 'RCC': 129, 'GYU': 130, 'RAC': 131, 'RGA': 132, 'GCS': 133, 'AAM': 134, 'CGY': 135, 'CVU': 136, 'AGK': 137, 'GKA': 138, 'AGR': 139, 'SUG': 140, 'UNA': 141, 'CMA': 142, 'CCK': 143, 'RGC': 144, 'YAA': 145, 'RGU': 146, 'ACY': 147, 'GGS': 148, 'ASC': 149, 'UYC': 150, 'YCU': 151, 'ABG': 152, 'UUD': 153, 'RAA': 154, 'CYC': 155, 'KCC': 156, 'YCG': 157, 'CCW': 158, 'GGR': 159, 'AAR': 160, 'RUC': 161, 'GUR': 162, 'KUU': 163, 'RCU': 164, 'GGK': 165, 'AYC': 166, 'GRG': 167, 'GUK': 168, 'UKA': 169, 'UCY': 170, 'YCC': 171, 'CCR': 172, 'UUS': 173, 'CYU': 174, 'AYG': 175, 'YGG': 176, 'USU': 177, 'AUY': 178, 'UYG': 179, 'SGA': 180, 'YUG': 181, 'GAY': 182, 'YGC': 183, 'MUG': 184, 'ARC': 185, 'GMC': 186, 'URG': 187, 'SCC': 188, 'GCY': 189, 'MGU': 190, 'CGR': 191, 'GGY': 192, 'GYC': 193, 'CRU': 194, 'YUC': 195, 'CCY': 196, 'AKG': 197, 'CSU': 198, 'ARG': 199, 'GAK': 200, 'GAR': 201, 'ARA': 202, 'GSC': 203, 'GRC': 204, 'RGG': 205, 'RCG': 206, 'CWC': 207, 'WCU': 208, 'CMC': 209, 'ARU': 210, 'AGY': 211, 'ACR': 212, 'GCW': 213, 'CYG': 214, 'UKG': 215, 'YUU': 216, 'YAU': 217, 'AMC': 218, 'MGG': 219, 'GCR': 220, 'GCM': 221, 'SCU': 222, 'YGA': 223, 'ACS': 224, 'GUY': 225, 'CAR': 226, 'WAC': 227, 'SUA': 228, 'UMG': 229, 'CUS': 230, 'GMG': 231, 'GGM': 232, 'CUR': 233, 'UCR': 234, 'AKU': 235, 'CGM': 236, 'CKC': 237, 'GWA': 238, 'UYA': 239, 'RAU': 240, 'SGG': 241, 'CCS': 242, 'MAC': 243, 'CAY': 244, 'MCA': 245, 'AYA': 246, 'CSG': 247, 'UGM': 248, 'UYU': 249, 'GYA': 250, 'YAC': 251, 'GCK': 252, 'CYA': 253, 'YGU': 254, 'AKC': 255, 'ASU': 256, 'UAY': 257, 'CRC': 258, 'KAC': 259, 'GYG': 260, 'URU': 261, 'UUM': 262, 'MGA': 263, 'CWU': 264, 'AUS': 265, 'UCM': 266, 'GAM': 267, 'GAS': 268, 'YUA': 269, 'WGC': 270, 'GRA': 271, 'RAG': 272, 'USN': 273, 'AMA': 274, 'ASG': 275, 'ACK': 276, 'UCK': 277, 'UGR': 278, 'GUM': 279, 'KAA': 280, 'GUS': 281, 'URC': 282, 'YCA': 283, 'UUR': 284, 'GRM': 285, 'UCW': 286, 'UGY': 287, 'GKG': 288, 'GKU': 289, 'CRA': 290, 'CSC': 291, 'AGW': 292, 'AAW': 293, 'RUA': 294, 'UCS': 295, 'CSA': 296, 'AGS': 297, 'MCG': 298, 'YAG': 299, 'SUC': 300, 'GRY': 301, 'RYU': 302, 'RUG': 303, 'SAG': 304, 'MUU': 305, 'CCM': 306, 'MCU': 307, 'RUU': 308, 'UUK': 309, 'RCA': 310, 'CAS': 311, 'GWG': 312, 'CWS': 313, 'KGC': 314, 'WCG': 315, 'GMU': 316, 'WCC': 317, 'CUY': 318, 'KGA': 319, 'KGU': 320, 'GMR': 321, 'MGN': 322, 'AMG': 323, 'GSG': 324, 'SGC': 325, 'KCA': 326, 'CWG': 327, 'AUW': 328, 'URA': 329, 'AAY': 330, 'GMA': 331, 'MAG': 332, 'SAA': 333, 'AYU': 334, 'UKK': 335, 'KGG': 336, 'MCC': 337, 'SAN': 338, 'ASA': 339, 'UAS': 340, 'SAU': 341, 'RNN': 342, 'MUA': 343, 'CMG': 344, 'CKG': 345, 'WCA': 346, 'RYC': 347, 'KUA': 348, 'UUY': 349, 'SAC': 350, 'CGS': 351, 'UGS': 352, 'UKY': 353, 'WAA': 354, 'ACM': 355, 'KUC': 356, 'UYY': 357, 'UWU': 358, 'CUK': 359, 'ACW': 360, 'CWA': 361, 'WGG': 362, 'SGU': 363, 'AAK': 364, 'WUA': 365, 'ARN': 366, 'GRU': 367, 'UAR': 368, 'UKC': 369, 'AWU': 370, 'UMC': 371, 'WAG': 372, 'UKU': 373, 'KUG': 374, 'UGK': 375, 'CAK': 376, 'MUC': 377, 'UAM': 378, 'MAU': 379, 'CKU': 380, 'UMU': 381, 'GWC': 382, 'KCG': 383, 'URY': 384, 'CYR': 385, 'AGM': 386, 'CSK': 387, 'GKC': 388, 'MNN': 389, 'WUC': 390, 'SUU': 391, 'USA': 392, 'MAA': 393, 'WUU': 394, 'KCU': 395, 'USC': 396, 'UMA': 397, 'UUW': 398, 'CUM': 399, 'GUW': 400, 'GSA': 401, 'GWU': 402, 'KAU': 403, 'SCG': 404, 'YKG': 405, 'UGW': 406, 'WSG': 407, 'WGA': 408, 'CAM': 409, 'USK': 410, 'USG': 411, 'AWG': 412, 'UAW': 413, 'UWC': 414, 'GYR': 415, 'WAU': 416, 'CWK': 417, 'WKU': 418, 'SSU': 419, 'AUR': 420, 'KAG': 421, 'UMK': 422, 'AAS': 423, 'CMU': 424, 'GWR': 425, 'GAW': 426, 'RCN': 427, 'AKA': 428, 'GMW': 429, 'MWG': 430, 'MCN': 431, 'CUW': 432, 'GGW': 433, 'CKA': 434, 'SGN': 435, 'KMC': 436, 'CYY': 437, 'WGN': 438, 'CGK': 439, 'YMG': 440, 'CYM': 441, 'UWA': 442, 'SWC': 443, 'GSR': 444, 'WGU': 445, 'UAK': 446, 'CYN': 447, 'GRR': 448, 'YCN': 449, 'AWC': 450, 'AYN': 451, 'VUA': 452, 'NCN': 453, 'DNN': 454, 'NCB': 455, 'NAN': 456, 'UWG': 457, 'DUG': 458, 'VAB': 459, 'WWG': 460, 'GSU': 461, 'CAW': 462, 'UYW': 463, 'WUG': 464, 'ASN': 465, 'RGN': 466, 'AUK': 467, 'CGW': 468, 'YRC': 469, 'AWA': 470, 'CKS': 471, 'SGS': 472}

init_token = '<sos>'
eos_token  = '<eos>'


def tokenize_seq(seq: str) -> list[str]:
    """
    Sliding-window trigram tokenisation for a UTR sequence.

    Produces overlapping trigrams for positions 0 … len(seq)-3 inclusive,
    wrapped in <sos> / <eos> sentinels.

    Example
    -------
    'AUGCAU'  →  ['<sos>', 'AUG', 'UGC', 'GCA', 'CAU', '<eos>']
    """
    seq = seq.replace("T", "U")          # ensure RNA alphabet
    tokens = [init_token]
    for i in range(len(seq) - 2):        # produces len(seq)-2 trigrams
        tokens.append(seq[i : i + 3])
    tokens.append(eos_token)
    return tokens


def numericalize(seq: str, vocab: dict) -> list[int]:
    """
    Convert a raw nucleotide sequence to a list of integer token indices.

    BUG FIX vs original: the old `numericalize` called `tokenize_seq` (which
    already prepends/appends special tokens) and then wrapped the result in
    *another* [sos] … [eos], producing doubled sentinels.  This version
    calls `tokenize_seq` exactly once.

    Unknown trigrams fall back to 0 (masked / ignored by the embedding).
    """
    tokens = tokenize_seq(seq)
    return [vocab.get(t, 0) for t in tokens]


def prepare_input(
    sequences: list[str],
    vocab: dict,
    device: str = 'cpu',
    pad_to = None,
) :
    """
    Encode a list of sequences into a padded LongTensor ready for the model.

    Parameters
    ----------
    sequences : list of raw nucleotide strings (DNA or RNA)
    vocab     : token → index mapping
    device    : torch device string
    pad_to    : if given, pad/truncate all sequences to this fixed length;
                otherwise pad to the length of the longest sequence in the batch

    Returns
    -------
    LongTensor of shape [B, T]
    """
    encoded = [numericalize(s, vocab) for s in sequences]
    T = pad_to if pad_to is not None else max(len(e) for e in encoded)
    padded = torch.zeros(len(encoded), T, dtype=torch.long)
    for i, e in enumerate(encoded):
        length = min(len(e), T)
        padded[i, :length] = torch.tensor(e[:length], dtype=torch.long)
    return padded.to(device)



# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from dataclasses import dataclass

    @dataclass
    class GEMORNA_5UTR_Config:
        block_size: int  = 768
        vocab_size: int  = 512   
        n_layer:    int  = 12
        n_head:     int  = 12
        n_embd:     int  = 144
        dropout:    float = 0.1
        bias:       bool  = True

    cfg = GEMORNA_5UTR_Config()

    model = adapted_GemoRNA(
        vocab_size  = cfg.vocab_size,
        n_embd      = cfg.n_embd,
        n_head      = cfg.n_head,
        dropout     = cfg.dropout,
        bias        = cfg.bias,
        block_size  = cfg.block_size,
        n_layer     = cfg.n_layer,
        num_classes = 1,          # scalar regression
        pooling     = 'last',
        freeze_backbone = True,   # fine-tune head only
    )

    # Load pretrained weights (comment out if you don't have the checkpoint yet)
    # model.load_pretrained_backbone('checkpoints/5utr.pt')

    # Dummy forward pass
    seqs = ['AUGCAUGCAUGCA', 'GCAUGCAUGCAUGCAU']
    x = prepare_input(seqs, five_prime_utr_vocab, device='cpu')
    print(f"Input shape:  {x.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")   # [2, 1]
    print(f"Predictions:  {out}")




#########################################################
#          FRAMEPOOL
########################################################

"""
PyTorch translation of framepool.py
Original: TensorFlow/Keras implementation of FramePool UTR model.

Key differences from the Keras version:
  - PyTorch Conv1d expects (batch, channels, seq_len); inputs/outputs are transposed
    at the model boundary so the public API still accepts (batch, seq_len, 4).
  - Masked average pooling and FrameSliceLayer logic are preserved exactly.
  - Residual skip connections use a 1x1 projection conv when channel sizes differ
    (e.g. on the first layer where in_channels != n_filters).
  - BatchNorm uses the channel dimension (dim=1 in PyTorch).
  - SpatialDropout1D is replicated with torch Dropout2d on the channel dim.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_pad_mask(x: torch.Tensor) -> torch.Tensor:
    """
    x: (batch, seq_len, 4)  — one-hot nucleotide encoding
    Returns mask: (batch, seq_len) — 1 where real sequence, 0 where padding
    """
    return x.sum(dim=2)  # non-zero rows are real nucleotides


def apply_pad_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    tensor: (batch, channels, seq_len)  [PyTorch conv layout]
    mask:   (batch, seq_len)
    """
    return tensor * mask.unsqueeze(1)  # broadcast over channel dim


def global_avg_pool_masked(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked global average pooling.
    tensor: (batch, channels, seq_len)
    mask:   (batch, seq_len)
    Returns: (batch, channels)
    """
    mask_expanded = mask.unsqueeze(1)                        # (batch, 1, seq_len)
    summed = (tensor * mask_expanded).sum(dim=2)             # (batch, channels)
    counts = mask_expanded.sum(dim=2).clamp(min=1e-9)        # (batch, 1)
    return summed / counts


# ---------------------------------------------------------------------------
# Custom layers
# ---------------------------------------------------------------------------

class LogNonhomogenousGeometric(nn.Module):
    """
    Log-probability layer for a non-homogeneous geometric distribution.
    Input x: any shape — operates element-wise then cumulates along dim=1.
    Preserves input shape.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_P = F.logsigmoid(x)                                    # log(sigmoid(x))
        log_inverse_P = -x + log_P                                 # log(1 - sigmoid(x))
        cumul_P = torch.cumsum(log_inverse_P, dim=1)               # cumulative sum
        # Exclusive cumsum: shift right by 1, pad with 0 at position 0
        cumul_P = F.pad(cumul_P[:, :-1], (1, 0), value=0.0)        # (batch, seq_len, ...)
        Q = log_P + cumul_P
        return Q


class FrameSliceLayer(nn.Module):
    """
    Slices a tensor into three interleaved frames (every 3rd position),
    anchored to the *end* of the sequence (start codon) by first reversing.

    Accepts tensors in PyTorch layout: (batch, channels, seq_len)
    or 2-D mask layout:               (batch, seq_len)
    Returns a list of three tensors of the same layout.
    """
    def forward(self, x: torch.Tensor):
        is_2d = (x.dim() == 2)
        seq_dim = 1 if is_2d else 2
        # Reverse along sequence dimension (anchor to start codon / fixed end)
        x = x.flip(dims=[seq_dim])
        seq_len = x.shape[seq_dim]
        idx0 = torch.arange(0, seq_len, 3, device=x.device)
        idx1 = torch.arange(1, seq_len, 3, device=x.device)
        idx2 = torch.arange(2, seq_len, 3, device=x.device)
        if is_2d:
            return [x[:, idx0], x[:, idx1], x[:, idx2]]
        else:
            return [x[:, :, idx0], x[:, :, idx1], x[:, :, idx2]]


# ---------------------------------------------------------------------------
# Building-block helpers
# ---------------------------------------------------------------------------

class ConvolveAndMask(nn.Module):
    """
    Single Conv1d + ReLU + optional BatchNorm + optional SpatialDropout.
    Operates in PyTorch layout (batch, channels, seq_len).
    """
    def __init__(self, in_channels: int, n_filters: int, kernel_size: int,
                 padding: str = "same", dilation: int = 1,
                 batchnorm: bool = False, conv_dropout: float = 0.0):
        super().__init__()
        self.padding_mode = padding
        if padding == "causal":
            # Manual left-padding so causality is respected
            self.pad_size = (kernel_size - 1) * dilation
            conv_pad = 0
        elif padding == "same":
            # Symmetric padding to keep seq_len unchanged
            self.pad_size = None
            conv_pad = ((kernel_size - 1) * dilation) // 2
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=conv_pad,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(n_filters) if batchnorm else None
        # SpatialDropout1D drops entire channels; nn.Dropout2d on (N,C,1,L) achieves this
        self.dropout = nn.Dropout2d(p=conv_dropout) if conv_dropout > 0.0 else None
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (batch, channels, seq_len)
        mask: (batch, seq_len)
        """
        if self.padding_mode == "causal":
            x = F.pad(x, (self.pad_size, 0))  # left-pad only
        x = self.conv(x)
        x = self.relu(x)
        x = apply_pad_mask(x, mask)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.dropout is not None:
            x = self.dropout(x.unsqueeze(-1)).squeeze(-1)  # spatial dropout trick
        return x


class InceptionBlock(nn.Module):
    """
    Parallel Conv1d branches with kernel sizes 3, 5, 7, concatenated on channel dim.
    n_filters: list/tuple of 3 ints — filters for each branch.
    """
    def __init__(self, in_channels: int, n_filters, padding: str = "same",
                 dilation: int = 1, batchnorm: bool = False, conv_dropout: float = 0.0):
        super().__init__()
        self.branch3 = ConvolveAndMask(in_channels, n_filters[0], 3, padding, dilation, batchnorm, conv_dropout)
        self.branch5 = ConvolveAndMask(in_channels, n_filters[1], 5, padding, dilation, batchnorm, conv_dropout)
        self.branch7 = ConvolveAndMask(in_channels, n_filters[2], 7, padding, dilation, batchnorm, conv_dropout)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch3(x, mask),
            self.branch5(x, mask),
            self.branch7(x, mask),
        ], dim=1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FramePoolModel(nn.Module):
    """
    PyTorch equivalent of create_frame_slice_model() in the original Keras code.

    Public interface (matching Keras):
      input:  (batch, seq_len, 4)   — one-hot nucleotide, 0-padded
      output: (batch, 1)            — MRL prediction (unscaled)

    Parameters
    ----------
    n_conv_layers   : number of convolutional layers
    kernel_size     : list of kernel sizes, one per layer
    n_filters       : number of Conv1d output filters (int, or list for inception)
    dilations       : dilation rate per layer
    padding         : "same" or "causal"
    use_batchnorm   : apply BatchNorm after each conv
    conv_dropout    : spatial dropout rate per layer (list)
    use_inception   : use InceptionBlock instead of single Conv1d
    skip_connections: "" | "residual" | "dense"
    n_dense_layers  : number of FC layers before output
    fc_neurons      : neuron counts per FC layer (list)
    fc_drop_rate    : dropout rate after each FC layer
    only_max_pool   : if True, skip masked average pooling
    """
    def __init__(
        self,
        n_conv_layers: int = 3,
        kernel_size=None,
        n_filters: int = 128,
        dilations=None,
        padding: str = "causal",
        use_batchnorm: bool = False,
        conv_dropout=None,
        use_inception: bool = False,
        skip_connections: str = "",
        n_dense_layers: int = 1,
        fc_neurons=None,
        fc_drop_rate: float = 0.2,
        only_max_pool: bool = False,
    ):
        super().__init__()
        if kernel_size is None:
            kernel_size = [8] * n_conv_layers
        if dilations is None:
            dilations = [1] * n_conv_layers
        if conv_dropout is None:
            conv_dropout = [0.0] * n_conv_layers
        if fc_neurons is None:
            fc_neurons = [64]
        self.skip_connections = skip_connections
        self.only_max_pool = only_max_pool
        self.n_conv_layers = n_conv_layers
        self.use_inception = use_inception
        # ------------------------------------------------------------------
        # Build convolutional stack
        # ------------------------------------------------------------------
        self.conv_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()  # 1×1 projections for residual shortcuts
        in_ch = 4  # one-hot nucleotide input channels
        for i in range(n_conv_layers):
            if use_inception:
                # n_filters must be a list of 3 ints for inception
                layer = InceptionBlock(in_ch, n_filters, padding, dilations[i],
                                       use_batchnorm, conv_dropout[i])
                out_ch = sum(n_filters)
            else:
                layer = ConvolveAndMask(in_ch, n_filters, kernel_size[i], padding,
                                        dilations[i], use_batchnorm, conv_dropout[i])
                out_ch = n_filters
            self.conv_layers.append(layer)
            # Projection for residual when channel sizes differ
            if skip_connections == "residual":
                if in_ch != out_ch:
                    self.proj_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False))
                else:
                    self.proj_layers.append(None)
            else:
                self.proj_layers.append(None)
            if skip_connections == "dense":
                in_ch = in_ch + out_ch  # dense concat grows channels
            else:
                in_ch = out_ch
        self.conv_out_channels = in_ch
        # ------------------------------------------------------------------
        # Frame slicing
        # ------------------------------------------------------------------
        self.frame_slice = FrameSliceLayer()
        # ------------------------------------------------------------------
        # Pooling → FC input size
        # 3 frames × (max_pool_channels [+ avg_pool_channels if not only_max_pool])
        # ------------------------------------------------------------------
        n_pool_types = 1 if only_max_pool else 2
        fc_in = 3 * n_pool_types * self.conv_out_channels
        # ------------------------------------------------------------------
        # Fully connected head
        # ------------------------------------------------------------------
        fc_layers = []
        for i in range(n_dense_layers):
            fc_layers.append(nn.Linear(fc_in, fc_neurons[i]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=fc_drop_rate))
            fc_in = fc_neurons[i]
        fc_layers.append(nn.Linear(fc_in, 1))
        self.fc = nn.Sequential(*fc_layers)
    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, 4)  — same convention as the Keras model
        Returns: (batch, 1)
        """
        # Pad mask from input (computed before transposing)
        x = x.permute(0, 2, 1)    
        pad_mask = compute_pad_mask(x)           # (batch, seq_len)
        # Transpose to PyTorch conv layout
        x = x.permute(0, 2, 1)                  # (batch, 4, seq_len)
        # ------------------------------------------------------------------
        # Convolutional stack
        # ------------------------------------------------------------------
        conv_features = x
        for i in range(self.n_conv_layers):
            shortcut = conv_features
            if self.use_inception:
                conv_features = self.conv_layers[i](conv_features, pad_mask)
            else:
                conv_features = self.conv_layers[i](conv_features, pad_mask)
            if self.skip_connections == "residual" and i > 0:
                proj = self.proj_layers[i]
                if proj is not None:
                    shortcut = proj(shortcut)
                conv_features = conv_features + shortcut
            elif self.skip_connections == "dense":
                conv_features = torch.cat([conv_features, shortcut], dim=1)
        # -----------------------------------------------------------------
        # Frame slicing  (still in conv layout: batch, channels, seq_len)
        # ------------------------------------------------------------------
        frame_features = self.frame_slice(conv_features)    # list of 3 × (batch, ch, seq_len//3)
        frame_masks = self.frame_slice(pad_mask)            # list of 3 × (batch, seq_len//3)
        # ------------------------------------------------------------------
        # Per-frame pooling
        # ------------------------------------------------------------------
        pooled = []
        for i in range(3):
            pooled.append(frame_features[i].max(dim=2).values)   # global max  (batch, ch)
        if not self.only_max_pool:
            for i in range(3):
                pooled.append(global_avg_pool_masked(frame_features[i], frame_masks[i]))
        concat_pooled = torch.cat(pooled, dim=1)   # (batch, 3 * n_pool_types * ch)
        # ------------------------------------------------------------------
        # FC head
        # ------------------------------------------------------------------
        out = self.fc(concat_pooled)               # (batch, 1)
        return out


# ---------------------------------------------------------------------------
# Factory helpers matching the original API
# ---------------------------------------------------------------------------

def create_frame_slice_model(
    n_conv_layers: int = 3,
    kernel_size=None,
    n_filters: int = 128,
    dilations=None,
    padding: str = "causal",
    use_batchnorm: bool = False,
    conv_dropout=None,
    use_inception: bool = False,
    skip_connections: str = "",
    n_dense_layers: int = 1,
    fc_neurons=None,
    fc_drop_rate: float = 0.2,
    only_max_pool: bool = False,
    loss: str = "mean_squared_error",          # kept for API compatibility; ignored
    use_counter_input: bool = False,           # not implemented (commented out in original)
    use_scaling_regression: bool = False,      # not implemented (commented out in original)
    library_size: int = 6,                     # not implemented (commented out in original)
) -> FramePoolModel:
    """
    Drop-in replacement for the Keras create_frame_slice_model().
    Returns a FramePoolModel (nn.Module).

    Note: `loss`, `use_counter_input`, `use_scaling_regression`, and
    `library_size` are accepted for signature compatibility but are not used,
    because those branches were commented out in the original Keras code.
    """
    if kernel_size is None:
        kernel_size = [8] * n_conv_layers
    if dilations is None:
        dilations = [1] * n_conv_layers
    if conv_dropout is None:
        conv_dropout = [0.0] * n_conv_layers
    if fc_neurons is None:
        fc_neurons = [64]
    return FramePoolModel(
        n_conv_layers=n_conv_layers,
        kernel_size=kernel_size,
        n_filters=n_filters,
        dilations=dilations,
        padding=padding,
        use_batchnorm=use_batchnorm,
        conv_dropout=conv_dropout,
        use_inception=use_inception,
        skip_connections=skip_connections,
        n_dense_layers=n_dense_layers,
        fc_neurons=fc_neurons,
        fc_drop_rate=fc_drop_rate,
        only_max_pool=only_max_pool,
    )


def load_framepool(path: str = "./../../models/utr_model_combined_residual_new.pt") -> FramePoolModel:
    """
    Load a saved FramePoolModel matching the default load_framepool() configuration.

    Expects a PyTorch state-dict saved with:
        torch.save(model.state_dict(), path)

    If you are migrating weights from the original Keras .h5 file you will need
    a separate weight-transfer script, as Keras and PyTorch store weights in
    different formats.
    """
    model = create_frame_slice_model(
        kernel_size=[7, 7, 7],
        only_max_pool=False,
        padding="same",
        skip_connections="residual",
        use_scaling_regression=True,   # accepted but no-op; see docstring above
        library_size=2,
    )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

