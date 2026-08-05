import unittest

from al_runner import _should_subsample_reweight_target


class ReweightTargetSubsampleTests(unittest.TestCase):
    def test_hard_reweighting_honors_requested_target_size(self):
        self.assertTrue(
            _should_subsample_reweight_target("hard", 50_000, 4_419_274)
        )

    def test_no_reweighting_does_not_build_target_subsample(self):
        self.assertFalse(
            _should_subsample_reweight_target("none", 50_000, 4_419_274)
        )

    def test_missing_or_oversized_request_uses_full_target(self):
        self.assertFalse(
            _should_subsample_reweight_target("hard", None, 4_419_274)
        )
        self.assertFalse(
            _should_subsample_reweight_target("hard", 4_419_274, 4_419_274)
        )


if __name__ == "__main__":
    unittest.main()
