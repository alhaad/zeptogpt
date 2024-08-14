import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MultiheadedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x): # (B, T, C=embed_dim)
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadedSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(approximate='tanh'), nn.Linear(4*embed_dim, embed_dim), nn.Dropout())
    
    def forward(self, inputs):
        outs = inputs + self.attn(self.ln1(inputs))
        outs = outs + self.mlp(self.ln2(outs))
        return outs

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_decoder_layers):
        super().__init__()
        self.block_size = block_size
        self.tok_emb_table = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb_table = nn.Embedding(block_size, embed_dim)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(embed_dim, num_heads) for _ in range(num_decoder_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, inputs): # inputs: (B, T)
        B, T = inputs.shape
        tok_embed = self.tok_emb_table(inputs) # (B, T) -> (B, T, C=embed_dim)
        pos_embed = self.pos_emb_table(torch.arange(T, device=inputs.device)) # (T, C=embed_dim)
        x = tok_embed + pos_embed  # (B, T, C)
        x = self.decoder_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, C) -> (B, T, vocab_size)
        return logits
    
    @torch.no_grad
    def generate(self, context, num_tokens): # context: (1, T)
        for _ in range(num_tokens):
            logits = self(context[:, -self.block_size:])[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
        return context
