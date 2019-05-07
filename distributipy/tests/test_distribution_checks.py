import unittest
import numpy as np
from distributipy.distribution_checks import *


class TestDistributionChecks(unittest.TestCase):

    def test_two_sided_normal_distribution_check(self):
        data = np.random.normal(size=100)
        self.assertFalse(is_not_normally_distributed(data))

    def test_one_sided_normal_distribution_check(self):
        data = np.random.normal(size=100)
        self.assertFalse(is_not_normally_distributed(data, alternative="less"))

    def test_normal_distribution_check_rejection(self):
        data = np.random.poisson(size=100)
        self.assertTrue(is_not_normally_distributed(data))

    def test_error_in_normal_distribution_check_on_invalid_alpha(self):
        data = np.random.normal(size=100)
        with self.assertRaises(TypeError):
            is_not_normally_distributed(data, alpha=-0.1)
        with self.assertRaises(TypeError):
            is_not_normally_distributed(data, alpha=1.1)
        with self.assertRaises(TypeError):
            is_not_normally_distributed(data, alternative="foobar")
        with self.assertRaises(TypeError):
            is_not_normally_distributed(data, mode="foobar")

    def test_error_in_normal_distribution_check_on_invalid_alternative(self):
        data = np.random.normal(size=100)
        with self.assertRaises(TypeError):
            is_not_normally_distributed(data, alternative="foobar")

    def test_error_in_normal_distribution_check_on_invalid_mode(self):
        data = np.random.normal(size=100)
        with self.assertRaises(TypeError):
            is_not_normally_distributed(data, mode="foobar")
