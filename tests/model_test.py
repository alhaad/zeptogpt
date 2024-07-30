from zeptogpt.model import CausalSelfAttentionHead, SimpleGPT

import torch
import torch.nn.functional as F
import unittest

class CausalSelfAttentionHeadTest(unittest.TestCase):
    def test_out_shape(self):
        sa = CausalSelfAttentionHead(32, 8)
        out = sa(torch.rand((4, 8, 32)))
        self.assertEqual(out.shape, (4, 8, 8))
    
    def test_zero(self):
        sa = CausalSelfAttentionHead(32, 8, bias=False)
        out = sa(torch.zeros((4, 8, 32)))
        self.assertEqual(out.shape, (4, 8, 8))
        self.assertTrue(torch.all(out == torch.zeros((4, 8, 8))))

    def test_scaled_dot_attention(self):
        q = torch.rand((4, 8, 8))
        k = torch.rand((4, 8, 8))
        v = torch.rand((4, 8, 8))

        out1 = CausalSelfAttentionHead.scaled_dot_product_attention(q, k, v, 'cpu')
        out2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        self.assertTrue(torch.all(torch.isclose(out1, out2)))

class SimpleGPTTest(unittest.TestCase):
    def test_out_shape(self):
        model = SimpleGPT(1000, 32, 8, 4, 2)
        out = model(torch.randint(0, 1000, (4, 8)))
        self.assertEqual(out.shape, (4, 8, 1000))

    def test_gen(self):
        model = SimpleGPT(1000, 32, 8, 4, 2)
        out = model.generate(torch.zeros((1, 1), dtype=torch.long), 10)
        self.assertEqual(out.shape, (1, 11))