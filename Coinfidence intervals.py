import math
import numpy as np

def confidence_intervals(data_line, bootstrap_resamples = 5000):
    """
    input:
    data_line: assumed to be n x t, where n is the number of evaluation runs and t is the number of evaluations periods

    output:
    lines: 3 x t, t is still the number of evaluation periods. First row has lower bound, second row has the mean, third row has the upper bound
    """

    N, T = data_line.shape

    lines = np.zeros((3, T))

    for t in range(T):
        resampled = np.random.choice(
            data_line[:, t],
            size=(bootstrap_resamples, N),
            replace=True
        )
        #print(resampled.shape)
        bootstrapped_means = resampled.mean(axis=1)
        #print(bootstrapped_means)
        #print(bootstrapped_means.shape)

        low, high = np.percentile(bootstrapped_means, [2.5, 97.5])
        #print(low, high)
        lines[0,t] = low
        lines[1,t] = np.mean(data_line[:,t])
        lines[2,t] = high

    return lines

data = np.random.randn(50, 20)
lines = confidence_intervals(data)
#print(lines)
