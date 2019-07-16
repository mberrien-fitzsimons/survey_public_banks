import numpy as np
# variance for a single sampe distribution
def singleVariance(sample):
    """
    calculates the variance of a sample distribution.

    calculates the sample variance as given by the formula large s^{2} = \frac{\sum_{i=1}^{n}(x_{i} - \bar{x})^{2}}{n-1}

    Parameters:
    sample (array, series): array or list of numbers. These numbers will be used to
    calculate the variance of the sample distribution.

    Returns:
    int: sample distribution variance

    """
    mean = sample.mean()
    var = sum((sample - mean)**2)/(len(sample) - 1)
    return var

# pooled variance for two distributions
def pooledVariance(samp1, samp2):
    """
    calculates the pooled variance of two sample distributions

    calculates the pooled sample variance as given by the formula large s^{2}{p} = \frac{(n{1} -1)s^{2}{1} + (n{2} -1)s^{2}{2}}{n{1} + n_{2} - 2}

    Parameters:
    sampl1 (array, series): array or list of numbers. These numbers will be used to calculate the pooled variance of the sample distribution.
    sampl2 (array, series): array or list of numbers. These numbers will be used to calculate the pooled variance of the sample distribution.

    Returns:
    int: pooled sample distribution variance

    """
    n_1 = len(samp1)
    n_2 = len(samp2)
    var_1 = singleVariance(samp1)
    var_2 = singleVariance(samp2)
    pooled_var = ((n_1 - 1)*var_1 + (n_2 - 1)*var_2)/ (n_1+n_2-2)
    return pooled_var

def ttest_two(samp1, samp2):
    """
    calculates the pooled variance of two sample distributions

    calculates the pooled sample variance as given by the formula large s^{2}{p} = \frac{(n{1} -1)s^{2}{1} + (n{2} -1)s^{2}{2}}{n{1} + n_{2} - 2}

    Parameters:
    sampl1 (array, series): array or list of numbers. These numbers will be used to calculate the pooled variance of the sample distribution.
    sampl2 (array, series): array or list of numbers. These numbers will be used to calculate the pooled variance of the sample distribution.

    Returns:
    int: pooled sample distribution variance

    """
    mean_1 = samp1.mean()
    mean_2 = samp2.mean()
    n_1 = len(samp1)
    n_2 = len(samp2)
    pool_var = pooledVariance(samp1, samp2)
    return (mean_1 - mean_2)/ np.sqrt(pool_var*((1/n_1)+(1/n_2)))
