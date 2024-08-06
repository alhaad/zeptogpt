from zeptogpt.model import MultiheadedSelfAttention, SimpleGPT

import torch
import torch.nn.functional as F
import unittest

class MultiheadedSelfAttentionTest(unittest.TestCase):
    def test_out_shape(self):
        sa = MultiheadedSelfAttention(32, 4)
        out = sa(torch.rand((4, 8, 32)))
        self.assertEqual(out.shape, (4, 8, 32))
    
    def test_zero(self):
        sa = MultiheadedSelfAttention(32, 8, bias=False)
        out = sa(torch.zeros((4, 8, 32)))
        self.assertEqual(out.shape, (4, 8, 32))
        self.assertTrue(torch.all(out == torch.zeros((4, 8, 32))))

class SimpleGPTTest(unittest.TestCase):
    def test_out_shape(self):
        model = SimpleGPT(1000, 32, 8, 4, 2)
        out = model(torch.randint(0, 1000, (4, 8)))
        self.assertEqual(out.shape, (4, 8, 1000))

    def test_gen(self):
        model = SimpleGPT(1000, 32, 8, 4, 2)
        out = model.generate(torch.zeros((1, 1), dtype=torch.long), 10)
        self.assertEqual(out.shape, (1, 11))