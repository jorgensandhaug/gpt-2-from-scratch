import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Attention(nn.Module):
    """
    This code is copied from https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
    """
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), num_heads=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_size, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Conv1d(config.hidden_size, config.hidden_size)
        self.c_proj = nn.Conv1d(config.hidden_size, config.hidden_size)
        self.act = nn.NewGELUActivation()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Config(object):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_position_embeddings, type_vocab_size, initializer_range):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range



class GPT2Model(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.config = GPT2Config(
            vocab_size=128,#50257,
            hidden_size=384,#768,
            num_layers=6,#12,
            num_heads=6,#,
            max_position_embeddings=512,#1024,
            type_vocab_size=2,
            initializer_range=0.02
        )
        
        self.model = nn.ModuleDict(
            wte=nn.Embedding(self.config.vocab_size, self.config.hidden_size),
            wpe=nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size),
            drop=nn.Dropout(p=0.1),
            h=nn.ModuleList([
                GPT2Block(self.config) for _ in range(self.config.num_layers)
            ]),
            ln_f=nn.LayerNorm(self.config.hidden_size, eps=1e-5),
        )
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)