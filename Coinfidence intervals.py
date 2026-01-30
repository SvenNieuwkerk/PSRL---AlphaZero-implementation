import math
import numpy as np
import pickle
import os
import ipywidgets

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

def extracts_from_traces(list_of_traces):
    """
    Assuming traces is a list of dictonairies
    """
    #list_of_traces is T long, where T is the number of evaluation periods
    T = len(list_of_traces)
    #Every dictonairy has N keys, where N is the number of evaluation runs
    N = len(list_of_traces[0].keys())
    for t, traces in enumerate(list_of_traces):
        for i, key in enumerate(traces.keys()):
            trace = traces[key]

            length_episode = len(trace["actions"])
            accumalitive_reward = trace["rewards"].sum()
            average_reward_per_step_episode = accumalitive_reward / length_episode
            mc_reward = trace["mc_return"].avg() #What is this
            success = trace["reward"][-1]>50
            collision = trace["reward"][-1]<50
            max_steps = not(success or collision)

            #store in data_matrix[i,t], #maybe a dictonairy that stores all datamatrices?

    #For all datamatrices
    #plot_data confidence_intervals(data_matrix)

    #Maybe store all plot data in one dictonairy


