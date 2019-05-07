import scipy.stats as st


def is_not_normally_distributed(data, alpha=0.05, alternative='two-sided', mode='approx', verbose=False):
    """
    Performs a Kolmogorov-Smirnov-Test for normal distribution. The tested hypothesis is that the data is not
    normally distributed. If the p-value is smaller than alpha, it returns True.

    :param data:                1-dimensional numpy array or pandas DataFrame of shape (m, 1).
    :param alpha:               Float. Defines the significance level.
    :param alternative:         String. Either 'two-sided', 'less' or 'greater'. Defines the alternative hypothesis.
    :param mode:                String. Either 'approx' or 'asymp'. See scipy.stats.kstest for more info.
    :param verbose:             True or False. True for verbose output.

    :return:                    True if data is not normally distributed.
                                False if alternative hypothesis cannot be rejected.

    """

    if alpha <= 0.0 or alpha >= 1:
        raise TypeError("Value for 'alpha' is {0}, but must be a value between 0 and 1.".format(alpha))

    if alternative not in ['two-sided', 'less', 'greater']:
        raise TypeError("Value for parameter 'alternative' must be either 'two-sided', 'less' or 'greater'.")

    if mode not in ['approx', 'asymp']:
        raise TypeError("Value for parameter 'mode' must be either 'approx' or 'asymp'.")

    # Test alternative hypothesis
    alternative_hypothesis = st.kstest(data, 'norm', alternative=alternative, mode=mode)

    # Compare the p-value with the given alpha and return the respective result
    if alternative_hypothesis.pvalue < alpha:
        if verbose:
            print("Not normally distributed with a p-value of {0}.".format(alternative_hypothesis.pvalue))
        return True
    elif alternative_hypothesis.pvalue >= alpha:
        if verbose:
            print("Normally distributed with a p-value of {0}.".format(alternative_hypothesis.pvalue))
        return False
    else:
        raise IOError("Did not get a p-value for the Kolmogorov-Smirnov-Test.")
