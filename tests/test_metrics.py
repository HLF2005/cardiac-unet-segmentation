import unittest

import torch

from src.metrics.segmentation import dice_score


class TestDiceScore(unittest.TestCase):
    def test_perfect_prediction(self):
        targets = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
            ]
        )

        score = dice_score(
            predictions=targets,
            targets=targets,
            num_classes=2,
        )

        self.assertEqual(score.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
