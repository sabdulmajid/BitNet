from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from train_bitdistill import SubLNLinear


class PromoteToFloat32(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


class SubLNDtypeContractTest(unittest.TestCase):
    def test_subln_output_preserves_projection_input_dtype(self) -> None:
        projection = nn.Linear(8, 3, bias=False, dtype=torch.bfloat16)
        module = SubLNLinear(projection, eps=1e-5)
        module.subln = PromoteToFloat32()

        x = torch.randn(2, 8, dtype=torch.bfloat16)
        output = module(x)

        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output, projection(x))


if __name__ == "__main__":
    unittest.main()
