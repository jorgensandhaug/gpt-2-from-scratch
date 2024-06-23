import torch
import torch.nn as nn
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # sort in descending order
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

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
        self.c_fc = nn.Linear(config.hidden_size, 4*config.hidden_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.hidden_size, config.hidden_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)  
        return x


class GPT2Config(object):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_position_embeddings):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings



class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.model = nn.ModuleDict({
            'wte': nn.Embedding(self.config.vocab_size, self.config.hidden_size),
            'wpe': nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size),
            'drop': nn.Dropout(p=0.1),
            'h': nn.ModuleList([
                GPT2Block(self.config) for _ in range(self.config.num_layers)
            ]),
            'ln_f': nn.LayerNorm(self.config.hidden_size, eps=1e-5),
        })
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids):
        # print(f"input_ids shape: {input_ids.shape}")
        # Get the input shape
        batch_size, seq_length = input_ids.size()
        
        # Generate position ids
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get word embeddings and position embeddings
        inputs_embeds = self.model['wte'](input_ids)
        position_embeds = self.model['wpe'](position_ids)
        
        # Add embeddings and apply dropout
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.model['drop'](hidden_states)

        
        # Pass through each transformer block
        for block in self.model['h']:
            hidden_states = block(hidden_states)
            
        
        # Apply final layer normalization
        hidden_states = self.model['ln_f'](hidden_states)
        
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # print(f"logits shape: {logits.shape}")
        
        return logits

    @classmethod
    def randomly_initialize_model(cls, config):
        model = cls(config)
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        return model

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        x = input_ids
        while x.shape[1] < max_length:
            # Forward pass through the model
            outputs = self(input_ids)
            
            # Get the logits for the next token prediction
            logits = outputs[:, -1, :] / temperature
            
            # Get the probs
            probs = F.softmax(logits, dim=-1)

            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)
        
        # Return the generated text
        return x[:, :]