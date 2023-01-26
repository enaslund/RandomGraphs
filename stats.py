from numpy import np
from scipy import stats
from TracyWidom import TracyWidom


def tracywidom_ks_test(x):
    """For an array x, returns the Kolmogorov-Smirnov-test for x normalized"""
    x = np.array(x)
    normed_list = sorted(list((x - np.mean(x)) / np.std(x)))
    tw1 = TracyWidom(beta=1)
    tw2 = TracyWidom(beta=2)
    tw4 = TracyWidom(beta=4)

    # The Tracy Widom distributions need to be made mean 0 std 1 for the ks-test.
    def tw1_cdf_adj(x):
        return tw1.cdf(np.sqrt(1.607781034581) * x - 1.2065335745820)

    def tw2_cdf_adj(x):
        return tw2.cdf(np.sqrt(0.8131947928329) * x - 1.771086807411)

    def tw4_cdf_adj(x):
        return tw4.cdf(np.sqrt(0.5177237207726) * x - 2.306884893241)

    return {
        "tw1": stats.kstest(normed_list, tw1_cdf_adj).statistic,
        "tw2": stats.kstest(normed_list, tw2_cdf_adj).statistic,
        "tw4": stats.kstest(normed_list, tw4_cdf_adj).statistic,
        "normal": stats.kstest(normed_list, stats.norm.cdf).statistic,
    }
