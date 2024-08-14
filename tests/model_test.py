from zeptogpt.model import MultiheadedSelfAttention, GPT

import jax
import jax.numpy as jnp
import unittest

class MultiheadedSelfAttentionTest(unittest.TestCase):
    def test_creation(self):
        key1, key2 = jax.random.split(jax.random.key(1337), num=2)
        inputs = jax.random.uniform(key1, (4, 8, 32), dtype=jnp.float32)
        model = MultiheadedSelfAttention(32, 8)

        params = model.init(key2, inputs)

        self.assertTrue('c_attn' in params['params'].keys())
        self.assertTrue('c_proj' in params['params'].keys())

    def test_out_shape(self):
        key1, key2 = jax.random.split(jax.random.key(1337), num=2)
        inputs = jax.random.uniform(key1, (4, 8, 32), dtype=jnp.float32)
        model = MultiheadedSelfAttention(32, 8)

        params = model.init(key2, inputs)
        out = model.apply(params, inputs)

        self.assertEqual(out.shape, (4, 8, 32))

    def test_attn(self):
        key1, key2 = jax.random.split(jax.random.key(1337), num=2)
        inputs = jax.random.uniform(key1, (1, 4, 2), dtype=jnp.float32)
        # Single headed attention
        model = MultiheadedSelfAttention(2, 1)

        params = model.init(key2, inputs)
        out = model.apply(params, inputs)

        # Reimplementation of single headed attention
        W, B = params['params']['c_attn']['kernel'], params['params']['c_attn']['bias']
        q_W, k_W, v_W = jnp.array_split(W, 3, axis=-1)
        q_B, k_B, v_B = jnp.array_split(B, 3, axis=-1)
        q = jnp.matmul(inputs, q_W) + q_B
        k = jnp.matmul(inputs, k_W) + k_B
        v = jnp.matmul(inputs, v_W) + v_B
        wei =  q @ k.swapaxes(-2, -1) / jnp.sqrt(2)
        tril = jnp.tril(jnp.ones((4, 4)))
        wei = jnp.where(tril == 0, float('-inf'), wei)
        wei =jax.nn.softmax(wei, axis=-1)
        y = wei @ v
        W, B = params['params']['c_proj']['kernel'], params['params']['c_proj']['bias']
        y = jnp.matmul(y, W) + B

        self.assertEqual(out.shape, (1, 4, 2))
        self.assertEqual(y.shape, (1, 4, 2))
        self.assertTrue(jnp.allclose(out, y))

class GPTTest(unittest.TestCase):
    def test_creation(self):
        key = jax.random.key(1337)
        inputs = jnp.zeros((4, 8), dtype=jnp.int32)
        model = GPT(1000, 8, 32, 4, 2)

        params = model.init(key, inputs)

        self.assertTrue('ln_f' in params['params'].keys())
        self.assertTrue('decoder_0' in params['params'].keys())
        self.assertTrue('decoder_1' in params['params'].keys())
        self.assertTrue('tok_embed' in params['params'].keys())
        self.assertTrue('pos_embed' in params['params'].keys())
    
    def test_out_shape(self):
        key = jax.random.key(1337)
        inputs = jnp.zeros((4, 8), dtype=jnp.int32)
        model = GPT(1000, 8, 32, 4, 2)

        params = model.init(key, inputs)
        out = model.apply(params, inputs)

        self.assertEqual(out.shape, (4, 8, 1000))
    
    def test_generation(self):
        key = jax.random.key(1337)
        inputs = jnp.zeros((1, 1), dtype=jnp.int32)
        model = GPT(1000, 8, 32, 4, 2)
        params = model.init(key, inputs)

        out = model.generate(key, params, inputs, 10)

        self.assertEqual(out.shape, (10, 1, 1))