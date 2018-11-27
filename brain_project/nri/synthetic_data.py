import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def sample_precision(num_nodes, edge_prob, seed=None, low_edge_prob=0.3, high_edge_prob=0.6):
    # Sample the graph, and get adjacency matrix
    G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=seed)
    if seed is not None:
        np.random.seed(seed)
    A = nx.to_numpy_array(G)

    # Modify the edges to have a value between `low_edge_prob` and `high_edge_prob`
    # either positive or negative.
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if A[i,j] != 1:
                continue
            new_val = (np.random.choice([1, -1]) *
                         (low_edge_prob +
                          (high_edge_prob - low_edge_prob) * np.random.rand()))
            # symmetric output matrix
            A[i,j] = new_val
            A[j,i] = new_val

    # Make the matrix positive definite
    eigs, _ = np.linalg.eig(A)
    lambda_min = min(eigs)
    print("lambda min: ", lambda_min)
    print("num edges: ", np.sum(A != 0) // 2)
    newA = A + (0.1 + abs(lambda_min)) * np.identity(A.shape[0])

    return newA


def sample_timeseries(precision_mat, mean=None, time_steps=100, lag_strength=0.0):
    """
    Args:
     - lag_strength : float
        We can use a 1-lag with strength `lag_strength`. If this is 0 then we have
        0-lag.
    """
    cov = np.linalg.inv(precision_mat)

    if mean is None:
        mean = np.zeros(cov.shape[0])

    tseries = [np.random.multivariate_normal(mean, cov)]
    for t in range(time_steps-1):
        new_t = np.random.multivariate_normal(mean, cov)
        tseries.append(lag_strength*tseries[-1] + new_t)

    #tseries = np.random.multivariate_normal(mean, cov, size=time_steps, check_valid="warn")

    return tseries


def gen_synthetic_tseries(num_clusters, num_tsteps, sample_size, num_nodes=5, edge_prob=0.2, seed=123912):

    low_edge_prob = 0.5
    high_edge_prob = 0.9
    lag_strength = 0.5

    precisions = [sample_precision(num_nodes,
                                   edge_prob,
                                   low_edge_prob=low_edge_prob,
                                   high_edge_prob=high_edge_prob,
                                   seed=seed+i)
                    for i in range(num_clusters)]

    tseries = [sample_timeseries(precisions[i],
                                 time_steps=num_tsteps,
                                 lag_strength=lag_strength)
                    for i in range(num_clusters)]

    # split according to sample size
    # in order to get samples of shape [num_nodes, sample_size], class_label
    all_samples = []
    all_labels = []
    for i, ts in enumerate(tseries):
        split_idxs = [sample_size * i for i in
            range(1, num_tsteps // sample_size + 1)]
        samples = [s.T for s in np.split(ts, split_idxs, axis=0)
                    if s.shape[0] == sample_size]
        all_samples.extend(samples)
        all_labels.extend([i] * len(samples))

    return all_samples, all_labels, precisions
