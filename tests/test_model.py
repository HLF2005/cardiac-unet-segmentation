import unittest

import torch

from src.models.unet import UNet


class TestUNet(unittest.TestCase):
    def test_output_shape(self):
        model = UNet(in_channels=1, out_channels=4)
        model.eval()

        images = torch.randn(1, 1, 32, 32)

        with torch.no_grad():
            predictions = model(images)

        self.assertEqual(
            predictions.shape,
            (1, 4, 32, 32),
        )


if __name__ == "__main__":
    unittest.main()
