import functools
import jax
import flax.linen as nn
import jax.numpy as jnp

class MultiheadedSelfAttention(nn.Module):
    n_embed: int
    n_heads: int

    @nn.compact
    def __call__(self, x): # (B, T, C=embed_dim)
        B, T, C = x.shape
        # key, query, value projections for all heads, but in a batch
        qkv = nn.Dense(3 * self.n_embed, name='c_attn')(x)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_heads, C // self.n_heads).swapaxes(1, 2) # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_heads, C // self.n_heads).swapaxes(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_heads, C // self.n_heads).swapaxes(1, 2) # (B, nh, T, hs)

        # y = nn.dot_product_attention(q, k, v, nn.make_causal_mask(jnp.ones((B, T))), deterministic=True) # flash attention
        att = jnp.einsum('...ij,...kj->...ik', q, k) * (1.0 / jnp.sqrt(C // self.n_heads)) # (B, nh, T, hs) x (B, nh, T, hs) -> (B, nh, T, T)
        mask = jnp.tril(jnp.ones((T, T)))
        att = jnp.where(mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        y = jnp.einsum('...ij,...jk->...ik', att, v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.swapaxes(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = nn.Dense(self.n_embed, name='c_proj')(y)
        return y
    
class MLP(nn.Module):
    n_embed: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.n_embed, name='c_fc')(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(self.n_embed, name='c_proj')(x)
        return x

class DecoderBlock(nn.Module):
    n_embed: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        x = x + MultiheadedSelfAttention(self.n_embed, self.n_heads, name='attn')(nn.LayerNorm(name='ln1')(x))
        x = x + MLP(self.n_embed, name='mlp')(nn.LayerNorm(name='ln2')(x))
        return x

class GPT(nn.Module):
    vocab_size: int
    block_size: int
    n_embed: int
    n_heads: int
    n_decoders: int

    # Cannot use @nn.compact to enable weight tying
    def setup(self):
        self.tok_embed = nn.Embed(self.vocab_size, self.n_embed)
        self.pos_embed = nn.Embed(self.block_size, self.n_embed)
        self.decoders = nn.Sequential([DecoderBlock(self.n_embed, self.n_heads, name='decoder_' + str(i)) for i in range(self.n_decoders)])
        self.ln_f = nn.LayerNorm()

    @nn.compact
    def __call__(self, x): # x.shape = (B, T)
        B, T = x.shape
        tok_embed = self.tok_embed(x)
        pos_embed = self.pos_embed(jnp.arange(0, T))
        x = tok_embed + pos_embed # (B, T, n_embed)
        x = self.decoders(x)
        x = self.ln_f(x)
        # x = nn.Dense(self.vocab_size, name='lm_head')(x) # (B, T, vocab_size)
        x = self.tok_embed.attend(x) # parameter sharing
        return x
    
    @functools.partial(jax.jit, static_argnames=("self", "length"))
    def generate(self, rng, params, init_context, length):
        def _scan_generate(carry, _):
            random_key, context = carry
            logits = self.apply(params, context)
            rng, rng_subkey = jax.random.split(random_key)
            new_token = jax.random.categorical(
                rng_subkey, logits[:, -1, :], axis=-1, shape=(1, 1)
            )
            context = jnp.concatenate([context[:, 1:], new_token], axis=1)
            return (rng, context), new_token

        _, new_tokens = jax.lax.scan(
            _scan_generate,
            (rng, init_context),
            (),
            length=length,
        )
        return new_tokens