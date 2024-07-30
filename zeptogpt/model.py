import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_size, bias=True):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_dim, head_size, bias)
        self.key = nn.Linear(embed_dim, head_size, bias)
        self.value = nn.Linear(embed_dim, head_size, bias)

    def forward(self, inputs): # (B, T, C)
        B, T, C = inputs.shape
        q = self.query(inputs) # (B, T, head_size)
        k = self.key(inputs) # (B, T, head_size)
        v = self.value(inputs)
        return CausalSelfAttentionHead.scaled_dot_product_attention(q, k, v, inputs.device)
    
    def scaled_dot_product_attention(q, k, v, device=None):
        attn = q @ k.transpose(-2, -1)
        attn = attn / (q.size(-1) ** 0.5) # Scaled attention
        causal_mask = torch.tril(torch.ones(attn.shape, device=device))
        attn = attn.masked_fill(causal_mask == 0, float('-inf')) # Masked attention
        attn = F.softmax(attn, dim=-1)
        return attn @ v

class MultiheadedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([CausalSelfAttentionHead(embed_dim, embed_dim // num_heads) for _ in range(num_heads)])

    def forward(self, inputs): # (B, T, C=embed_dim)
        outputs = torch.cat([h(inputs) for h in self.heads], dim=-1)
        return outputs

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multi_headed_attn = MultiheadedSelfAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.ReLU(), nn.Linear(4*embed_dim, embed_dim), nn.Dropout())
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, inputs):
        outs = inputs + self.multi_headed_attn(self.ln1(inputs))
        outs = outs + self.feed_forward(self.ln2(outs))
        return outs

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_decoder_layers):
        super().__init__()
        self.block_size = block_size
        self.tok_emb_table = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb_table = nn.Embedding(block_size, embed_dim)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(embed_dim, num_heads) for _ in range(num_decoder_layers)])
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, inputs): # inputs: (B, T)
        B, T = inputs.shape
        tok_embed = self.tok_emb_table(inputs) # (B, T) -> (B, T, C=embed_dim)
        pos_embed = self.pos_emb_table(torch.arange(T, device=inputs.device)) # (T, C=embed_dim)
        x = tok_embed + pos_embed  # (B, T, C)
        x = self.decoder_blocks(x)
        logits = self.lm_head(x) # (B, T, C) -> (B, T, vocab_size)
        return logits
    
    @torch.no_grad
    def generate(self, context, num_tokens): # context: (1, T)
        for _ in range(num_tokens):
            logits = self(context[:, -self.block_size:])[:,-1,:]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
        return context

sample_model = SimpleGPT(1000, 32, 8, 4, 2)
print(sample_model.generate(torch.zeros((1,1), dtype=torch.long), 10)[0].tolist())