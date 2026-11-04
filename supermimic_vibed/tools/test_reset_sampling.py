import unittest

import torch

from supermimic.reset_sampling import update_bin_returns


class ResetSamplingTest(unittest.TestCase):
    def test_duplicate_bins_average_returns_before_ema(self):
        bin_G = torch.tensor([1.0, 2.0, 3.0])
        bins = torch.tensor([1, 1, 1, 0])
        returns = torch.tensor([4.0, 2.0, 6.0, 5.0])

        update_bin_returns(bin_G, bins, returns)

        torch.testing.assert_close(bin_G, torch.tensor([1.8, 2.4, 3.0]))

    def test_update_is_independent_of_duplicate_order(self):
        bins = torch.tensor([0, 0, 0])
        returns = torch.tensor([1.0, 2.0, 7.0])
        first = update_bin_returns(torch.zeros(2), bins, returns)
        second = update_bin_returns(torch.zeros(2), bins, returns.flip(0))

        torch.testing.assert_close(first, second)

    def test_empty_batch_leaves_returns_unchanged(self):
        bin_G = torch.tensor([1.0, 2.0])
        updated = update_bin_returns(bin_G, torch.empty(0, dtype=torch.long), torch.empty(0))

        self.assertIs(updated, bin_G)
        torch.testing.assert_close(bin_G, torch.tensor([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
