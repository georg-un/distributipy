import unittest
import numpy as np
import scipy.stats as st
from distributipy.distribution_fitter import *


class TestDistributionFitter(unittest.TestCase):

    def test_fit_single_distribution(self):
        data = np.random.normal(loc=250, scale=5, size=500)
        distribution_fitter = DistributionFitter(data=data, n_bins=250)
        fitted_distribution = distribution_fitter.fit(st.norm)

        self.assertIsNotNone(fitted_distribution)
        self.assertEquals(fitted_distribution.distribution, st.norm)
        self.assertAlmostEqual(fitted_distribution.standard_deviation, 5, delta=5)
        self.assertAlmostEqual(fitted_distribution.mean, 250, delta=20)
        self.assertEquals(len(fitted_distribution.arg), 0)
        self.assertNotEquals(fitted_distribution.sse, 0.0)

    def test_normal_distribution_probability_x_less(self):
        data = np.random.normal(loc=250, scale=5, size=500)
        distribution_fitter = DistributionFitter(data=data, n_bins=250)
        fitted_distribution = distribution_fitter.fit(st.norm)

        self.assertAlmostEquals(fitted_distribution.probability_x_less_equal(250), 0.5, delta=10)
        self.assertGreater(fitted_distribution.probability_x_less_equal(300), 0.95)
        self.assertLess(fitted_distribution.probability_x_less_equal(200), 0.05)

    def test_laplace_distribution_probability_x_less(self):
        data = np.random.laplace(loc=250, scale=5, size=500)
        distribution_fitter = DistributionFitter(data=data, n_bins=250)
        fitted_distribution = distribution_fitter.fit(st.norm)

        self.assertAlmostEquals(fitted_distribution.probability_x_less_equal(250), 0.5, delta=10)
        self.assertGreater(fitted_distribution.probability_x_less_equal(300), 0.95)
        self.assertLess(fitted_distribution.probability_x_less_equal(200), 0.05)

    def test_normal_distribution_probability_x_greater(self):
        data = np.random.normal(loc=250, scale=5, size=500)
        distribution_fitter = DistributionFitter(data=data, n_bins=250)
        fitted_distribution = distribution_fitter.fit(st.norm)

        self.assertAlmostEquals(fitted_distribution.probability_x_greater_equal(250), 0.5, delta=10)
        self.assertLess(fitted_distribution.probability_x_greater_equal(300), 0.05)
        self.assertGreater(fitted_distribution.probability_x_greater_equal(200), 0.95)

    def test_normal_distribution_probability_for_x(self):
        data = np.random.normal(loc=250, scale=5, size=500)
        distribution_fitter = DistributionFitter(data=data, n_bins=250)
        fitted_distribution = distribution_fitter.fit(st.norm)

        self.assertAlmostEqual(fitted_distribution.value_for_probability_x(0.5), 250, delta=10)

    def test_find_single_best_fitting_distribution(self):
        data = np.random.normal(loc=250, scale=5, size=500)
        distribution_fitter = DistributionFitter(data=data, n_bins=250)
        fitted_distributions = distribution_fitter.best_n_fitting(1)

        self.assertIsNotNone(fitted_distributions.distributions)
        self.assertEquals(len(fitted_distributions.distributions), 1)

    def test_find_n_best_fitting_distributions(self):
        data = np.random.normal(loc=250, scale=5, size=500)
        distribution_fitter = DistributionFitter(data=data, n_bins=250)
        fitted_distributions = distribution_fitter.best_n_fitting(89)

        self.assertIsNotNone(fitted_distributions.distributions)
        self.assertAlmostEqual(len(fitted_distributions.distributions), 89, delta=5)
