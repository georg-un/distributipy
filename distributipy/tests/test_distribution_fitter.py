import unittest
import os

from distributipy.distribution_fitter import *

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TestDistributionFitter(unittest.TestCase):

    def test_fit_single_distribution(self):
        distribution_fitter = DistributionFitter(data=data, n_bins=125)
        fitted_distribution = distribution_fitter.fit(st.norm)
        data = np.genfromtxt(os.path.join(data_path, "normal.csv"), delimiter=",")

        self.assertIsNotNone(fitted_distribution)
        self.assertEqual(fitted_distribution.distribution, st.norm)
        self.assertAlmostEqual(fitted_distribution.standard_deviation, 5, delta=5)
        self.assertAlmostEqual(fitted_distribution.mean, 250, delta=20)
        self.assertEqual(len(fitted_distribution.arg), 0)
        self.assertNotEqual(fitted_distribution.sse, 0.0)

    def test_normal_distribution_probability_x_less(self):
        distribution_fitter = DistributionFitter(data=data, n_bins=125)
        fitted_distribution = distribution_fitter.fit(st.norm)
        data = np.genfromtxt(os.path.join(data_path, "normal.csv"), delimiter=",")

        self.assertAlmostEqual(fitted_distribution.probability_x_less_equal(250), 0.5, delta=10)
        self.assertGreater(fitted_distribution.probability_x_less_equal(300), 0.95)
        self.assertLess(fitted_distribution.probability_x_less_equal(200), 0.05)

    def test_laplace_distribution_probability_x_less(self):
        distribution_fitter = DistributionFitter(data=data, n_bins=125)
        fitted_distribution = distribution_fitter.fit(st.norm)
        data = np.genfromtxt(os.path.join(data_path, "laplace.csv"), delimiter=",")

        self.assertAlmostEqual(fitted_distribution.probability_x_less_equal(250), 0.5, delta=10)
        self.assertGreater(fitted_distribution.probability_x_less_equal(300), 0.95)
        self.assertLess(fitted_distribution.probability_x_less_equal(200), 0.05)

    def test_normal_distribution_probability_x_greater(self):
        distribution_fitter = DistributionFitter(data=data, n_bins=125)
        fitted_distribution = distribution_fitter.fit(st.norm)
        data = np.genfromtxt(os.path.join(data_path, "normal.csv"), delimiter=",")

        self.assertAlmostEqual(fitted_distribution.probability_x_greater_equal(250), 0.5, delta=10)
        self.assertLess(fitted_distribution.probability_x_greater_equal(300), 0.05)
        self.assertGreater(fitted_distribution.probability_x_greater_equal(200), 0.95)

    def test_normal_distribution_probability_for_x(self):
        distribution_fitter = DistributionFitter(data=data, n_bins=125)
        fitted_distribution = distribution_fitter.fit(st.norm)
        data = np.genfromtxt(os.path.join(data_path, "normal.csv"), delimiter=",")

        self.assertAlmostEqual(fitted_distribution.value_for_probability_x(0.5), 250, delta=10)

    def test_find_single_best_fitting_distribution(self):
        distribution_fitter = DistributionFitter(data=data, n_bins=125)
        fitted_distributions = distribution_fitter.best_n_fitting(1)
        data = np.genfromtxt(os.path.join(data_path, "normal.csv"), delimiter=",")

        self.assertIsNotNone(fitted_distributions.distributions)
        self.assertEqual(len(fitted_distributions.distributions), 1)

    def test_find_n_best_fitting_distributions(self):
        distribution_fitter = DistributionFitter(data=data, n_bins=125)
        fitted_distributions = distribution_fitter.best_n_fitting(89)
        data = np.genfromtxt(os.path.join(data_path, "normal.csv"), delimiter=",")

        self.assertIsNotNone(fitted_distributions.distributions)
        self.assertAlmostEqual(len(fitted_distributions.distributions), 89, delta=5)


if __name__ == '__main__':
    unittest.main()
