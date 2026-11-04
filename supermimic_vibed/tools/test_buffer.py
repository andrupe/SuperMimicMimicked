import unittest

import torch

from supermimic.buffer import FIELDS, ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def test_short_buffer_has_clear_error(self):
        buffer = ReplayBuffer(T=16, N=2, device="cpu")
        with self.assertRaisesRegex(ValueError, "8-step windows from 0 buffered steps"):
            buffer.sample_windows(B=1, k=8)

    def test_exact_window_length_is_valid(self):
        buffer = ReplayBuffer(T=8, N=2, device="cpu")
        transition = {}
        for key, (shape, dtype) in FIELDS.items():
            full_shape = (2,) + ((shape,) if isinstance(shape, int) else shape)
            transition[key] = torch.zeros(full_shape, dtype=dtype)
        for _ in range(8):
            buffer.add(transition)

        window = buffer.sample_windows(B=3, k=8)

        self.assertEqual(window["s"].shape, (3, 8, 71))
        self.assertEqual(window["mask"].shape, (3, 8))
        self.assertTrue(window["mask"].all())


if __name__ == "__main__":
    unittest.main()
