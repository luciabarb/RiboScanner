import torch
import torch.nn as nn
from torch.nn import functional as F
import math

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
            tril = tril.view(1, 1, block_size, block_size)
            self.register_buffer("causal_mask", tril)
    def forward(self, hidden_states):
        batch_sz, seq_len, emb_dim = hidden_states.size()
        proj = self.c_attn(hidden_states)
        q_proj, k_proj, v_proj = proj.chunk(3, dim=-1)
        head_dim = emb_dim // self.n_heads
        def shape_proj(tensor):
            return tensor.view(batch_sz, seq_len, self.n_heads, head_dim).transpose(1, 2)
        queries = shape_proj(q_proj)
        keys = shape_proj(k_proj)
        values = shape_proj(v_proj)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=0, is_causal=True)
        else:
            scaling = 1.0 / math.sqrt(head_dim)
            sim_matrix = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            sim_matrix = sim_matrix.masked_fill(mask == 0, float('-inf'))
            attention = F.softmax(sim_matrix, dim=-1)
            attention = self.attention_dropout(attention)
            y = torch.matmul(attention, values)
        y = y.transpose(1, 2).contiguous().reshape(batch_sz, seq_len, emb_dim)
        x = self.c_proj(y)
        x = self.residual_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = Attention(n_head, n_embd, dropout, bias, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)
    def forward(self, x):
        x1 = self.ln_1(x)
        attn_output = self.attn(x1)
        x = x + attn_output
        x2 = self.ln_2(x)
        mlp_output = self.mlp(x2)
        x = x + mlp_output
        return x


class UTR_(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, dropout, bias, block_size, n_layer):
        super().__init__()
        # IMPORTANT: use nn.Module, NOT ModuleDict
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab_size, n_embd)
        self.transformer.wpe = nn.Embedding(block_size, n_embd)
        self.transformer.drop = nn.Dropout(dropout)
        self.transformer.h = nn.ModuleList([
            DecoderBlock(n_embd, n_head, dropout, bias, block_size)
            for _ in range(n_layer)
        ])
        self.transformer.ln_f = LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying (correct and safe way)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        MEAN, STD = 0.0, 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=MEAN, std=STD)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=MEAN, std=STD)
    def forward(self, input_ids, targets=None):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        pos_indices = torch.arange(seq_len, device=device, dtype=torch.long)
        token_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(pos_indices)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.transformer.drop(hidden_states)
        for block in self.transformer.h:
            hidden_states = block(hidden_states)
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states[:, [-1], :])
        return logits, None


class adapted_GemoRNA(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, dropout, bias, block_size, n_layer, num_classes=1):
        super().__init__()
        self.backbone = UTR_(vocab_size, n_embd, n_head, dropout, bias, block_size, n_layer)
        # remove LM head influence
        self.backbone.lm_head = nn.Identity()
        self.classifier = nn.Linear(n_embd, num_classes)
    def forward(self, idx):
        # replicate UTR_ forward BUT stop before lm_head
        b, t = idx.size()
        tok_emb = self.backbone.transformer.wte(idx)
        pos_emb = self.backbone.transformer.wpe(torch.arange(t, device=idx.device))
        x = self.backbone.transformer.drop(tok_emb + pos_emb)
        for block in self.backbone.transformer.h:
            x = block(x)
        x = self.backbone.transformer.ln_f(x)
        # 👇 choose pooling strategy
        x = x[:, -1, :]   # last token (causal model)
        logits = self.classifier(x)
        return logits
