#Welch's Test to compare avgVol/MarcetCap at two different times, using 10-day and 30-day avergages:
import scipy.stats as stats
import numpy as np

# welch's t-test
def welch_ttest(samp1, samp2):
    """
    welches t-test.

    Compares the means of a variable from TWO groups

    Parameters:
    samp1 (array): array of numbers
    samp2 (array): array of numbers

    Returns:
    int: returns critical t-test value

    """
    num1 = samp1.mean() - samp2.mean()
    denom1 = np.sqrt(samp1.var(ddof = 1)/len(samp1) + samp2.var(ddof =1)/len(samp2))
    welch_t = np.abs(num1/denom1)
    return welch_t

# compute the effective degrees of freedom
def effective_dof(samp1, samp2):
    """
    Calculates effective degrees of freedom.

    Calculates effective degrees of freedom from two sample distributions

    Parameters:
    samp1 (array): array of numbers
    samp2 (array): array of numbers

    Returns:
    int: effective degrees of freedom

    """
    s1 = samp1.var(ddof=1)
    s2 = samp2.var(ddof=1)
    n1 = len(samp1)
    n2 = len(samp2)
    num2 = (s1/n1 + s2/n2)**2
    denom2 = (s1/n1)**2/(n1-1)+(s2/n2)**2/(n2-1)
    welch_dof = num2/denom2
    return welch_dof

# compute p value
def compute_pval(welch_t, welch_dof):
    """
    computes p-value

    computes the p-value of critical t-test value.

    Parameters:
    welch_t (int): critical t-statistic obtained from a t-test
    welch_dof (int): degrees of freedom

    Returns:
    int: p-value

    """
    p_value = 1-stats.t.cdf(welch_t, welch_dof)
    return p_value
