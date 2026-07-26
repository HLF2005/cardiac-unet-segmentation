import unittest

import torch

from src.config import DATA_DIR
from src.data.dataset import CardiacDataset


class TestCardiacDataset(unittest.TestCase):
    @unittest.skipUnless(
        (DATA_DIR / "patient001").exists(),
        "Dataset non disponible.",
    )
    def test_shapes_and_labels(self):
        dataset = CardiacDataset(
            [DATA_DIR / "patient001"],
            augmentation=True,
        )
        image, mask = dataset[5]

        self.assertEqual(image.shape, (1, 256, 256))
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.int64)
        self.assertTrue(
            set(torch.unique(mask).tolist()).issubset(
                {0, 1, 2, 3}
            )
        )


if __name__ == "__main__":
    unittest.main()
