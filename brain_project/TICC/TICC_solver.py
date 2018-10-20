import numpy as np
import math, time, collections, os, errno, sys, code, random
import matplotlib
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
from multiprocessing import Pool

from .src.TICC_helper import *
from .src.admm_solver import ADMMSolver

import os
from datetime import datetime

def time_str():
    now = datetime.now()
    return now.strftime("[%m-%d %H:%M:%S]")

def mkdir_p(path):
    """Recursively create directories."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class TICC:
    def __init__(self, window_size=10, number_of_clusters=5, lambda_parameter=11e-2,
                 beta=400, maxIters=1000, threshold=2e-5, write_out_file=False,
                 prefix_string="", num_proc=1, compute_BIC=False, cluster_reassignment=20):
        """
        Parameters:
            - window_size: size of the sliding window
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - write_out_file: (bool) if true, prefix_string is output file dir
            - prefix_string: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
        """
        self.window_size = window_size
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1

        self.model = None

    def fit(self, input_ts):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        # Get data into proper format
        times_series_arr, time_series_rows_size, time_series_col_size = self.load_data(input_ts)
        self.num_features = time_series_col_size

        ############
        # The basic folder to be created
        out_path = None
        if self.write_out_file:
            out_path = self.prepare_out_directory()

        # Train test split
        training_indices = getTrainTestSplit(time_series_rows_size, self.num_blocks,
                                             self.window_size)  # indices of the training samples
        num_train_points = len(training_indices)

        # Stack the training data
        complete_D_train = self.stack_training_data(times_series_arr, time_series_col_size, num_train_points,
                                                    training_indices)

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.number_of_clusters, covariance_type="full")
        gmm.fit(complete_D_train)
        clustered_points = gmm.predict(complete_D_train)
        gmm_clustered_pts = clustered_points + 0
        # K-means
        kmeans = KMeans(n_clusters=self.number_of_clusters).fit(complete_D_train)
        kmeans_clustered_pts = kmeans.labels_

        old_clustered_points = None  # points from last iteration

        # PERFORM TRAINING ITERATIONS
        try:
            pool = Pool(processes=self.num_proc)  # multi-threading
            for iters in range(self.maxIters):
                print("\n\n%s Iteration %d" % (time_str(), iters))
                # Get the train and test points
                train_clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
                for point, cluster_num in enumerate(clustered_points):
                    train_clusters_arr[cluster_num].append(point)

                len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

                # train_clusters holds the indices in complete_D_train for each of the clusters
                self.model = self.train_clusters(complete_D_train, time_series_col_size, pool,
                                            train_clusters_arr)

                self.model.update(self.optimize_clusters(self.model["optimized_results"]))

                clustered_points = self.predict_clusters(self.model, train_data=complete_D_train)

                # recalculate lengths
                new_train_clusters = collections.defaultdict(list) # {cluster: [point indices]}
                for point, cluster in enumerate(clustered_points):
                    new_train_clusters[cluster].append(point)

                before_empty_cluster_assign = clustered_points.copy()

                if iters != -1:
                    old_computed_covariance = self.model["computed_covariance"]
                    cluster_norms = [(np.linalg.norm(old_computed_covariance[i]), i) for i in
                                     range(self.number_of_clusters)]
                    norms_sorted = sorted(cluster_norms, reverse=True)
                    # clusters that are not 0 as sorted by norm
                    valid_clusters = [cp[1] for cp in norms_sorted if len(new_train_clusters[cp[1]]) != 0]

                    # Add a point to the empty clusters
                    # assuming more non empty clusters than empty ones
                    counter = 0
                    for cluster_num in range(self.number_of_clusters):
                        if len(new_train_clusters[cluster_num]) == 0:
                            cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                            counter = (counter + 1) % len(valid_clusters)
                            print("%s Cluster %d has 0 samples" % (time_str(), cluster_num))
                            start_point = np.random.choice(
                                new_train_clusters[cluster_selected])  # random point number from that cluster
                            for i in range(0, self.cluster_reassignment):
                                # put cluster_reassignment points from point_num in this cluster
                                point_to_move = start_point + i
                                if point_to_move >= len(clustered_points):
                                    break
                                clustered_points[point_to_move] = cluster_num
                                self.model["computed_covariance"][cluster_num] = old_computed_covariance[cluster_selected]
                                self.model["stacked_mean"][cluster_num] = complete_D_train[point_to_move, :]
                                self.model["end_mean"][cluster_num] = complete_D_train[point_to_move, :][
                                      (self.window_size - 1) * time_series_col_size:]

                for cluster_num in range(self.number_of_clusters):
                    print("%s Length of cluster %d: %f" % (time_str(), cluster_num,
                         sum([x == cluster_num for x in clustered_points])))

                self.write_plot(clustered_points, out_path, training_indices)

                # TEST SETS STUFF
                # LLE + swtiching_penalty
                # Segment length
                # Create the F1 score from the graphs from k-means and GMM
                # Get the train and test points
                if False:
                    train_confusion_matrix_EM = compute_confusion_matrix(self.number_of_clusters, clustered_points,
                                                                         training_indices)
                    train_confusion_matrix_GMM = compute_confusion_matrix(self.number_of_clusters, gmm_clustered_pts,
                                                                          training_indices)
                    train_confusion_matrix_kmeans = compute_confusion_matrix(self.number_of_clusters, kmeans_clustered_pts,
                                                                             training_indices)
                    ###compute the matchings
                    matching_EM, matching_GMM, matching_Kmeans = self.compute_matches(train_confusion_matrix_EM,
                                                                                      train_confusion_matrix_GMM,
                                                                                      train_confusion_matrix_kmeans)

                print("\n\n\n")

                if np.array_equal(old_clustered_points, clustered_points):
                    print("%s Training converged, breaking early." % (time_str()))
                    break
                old_clustered_points = before_empty_cluster_assign
                # end of training
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        if False:
            train_confusion_matrix_EM = compute_confusion_matrix(self.number_of_clusters, clustered_points,
                                                                 training_indices)
            train_confusion_matrix_GMM = compute_confusion_matrix(self.number_of_clusters, gmm_clustered_pts,
                                                                  training_indices)
            train_confusion_matrix_kmeans = compute_confusion_matrix(self.number_of_clusters, kmeans_clustered_pts,
                                                                     training_indices)

            self.compute_f_score(matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                                 train_confusion_matrix_GMM, train_confusion_matrix_kmeans)

        if self.compute_BIC:
            bic = computeBIC(self.number_of_clusters, time_series_rows_size, clustered_points, self.model["inverse_covariance"],
                             self.model["empirical_covariances"])
            return clustered_points, self.model["inverse_covariance"], bic

        return clustered_points, self.model["inverse_covariance"]

    def compute_f_score(self, matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
        train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        f1_EM_tr = -1  # computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
        f1_GMM_tr = -1  # computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
        f1_kmeans_tr = -1  # computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)
        print("\n\n")
        print("TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.number_of_clusters):
            matched_cluster__e_m = matching_EM[cluster]
            matched_cluster__g_m_m = matching_GMM[cluster]
            matched_cluster__k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster__e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster__g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster__k_means]

    def compute_matches(self, train_confusion_matrix_EM, train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        matching_Kmeans = find_matching(train_confusion_matrix_kmeans)
        matching_GMM = find_matching(train_confusion_matrix_GMM)
        matching_EM = find_matching(train_confusion_matrix_EM)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.number_of_clusters):
            matched_cluster_e_m = matching_EM[cluster]
            matched_cluster_g_m_m = matching_GMM[cluster]
            matched_cluster_k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster_e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster_g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster_k_means]
        return matching_EM, matching_GMM, matching_Kmeans

    def write_plot(self, clustered_points, out_path, training_indices):
        # Save a figure of segmentation
        if False:
            plt.figure()
            plt.plot(training_indices[0:len(clustered_points)], clustered_points, color="r")  # ,marker = ".",s =100)
            plt.ylim((-0.5, self.number_of_clusters + 0.5))
            if self.write_out_file:
                fname = os.path.join(out_path, "clusters_switch_pen=%.1f.png" % self.switch_penalty)
                plt.savefig(fname)
                print("%s Saved figure to %s" % (time_str(), fname))
                plt.close("all")
            ## Needed for matplotlib in notebook
            plt.pause(1)

    def smoothen_clusters(self, model, complete_D_train, n):
        LLE_all_points_clusters = np.zeros([complete_D_train.shape[0], self.number_of_clusters])
        for cluster in range(self.number_of_clusters):
            stacked_mean = model["stacked_mean"][cluster]
            inv_cov_matrix = model["inverse_covariance"][cluster]
            log_det_cov = model["log_det"][cluster]

            x = complete_D_train - stacked_mean # N x nw
            lle = (x.dot(inv_cov_matrix)*x).sum(axis=1)
            lle = lle + log_det_cov
            LLE_all_points_clusters[:, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(self, opt_res):
        log_det = {}
        computed_covariance = {}
        inverse_covariance = {}

        for cluster in range(self.number_of_clusters):
            if opt_res[cluster] == None:
                continue
            # Get the future result
            val = opt_res[cluster].get()
            print("%s Finished ADMM for cluster %d" % (time_str(), cluster))
            # THIS IS THE SOLUTION
            S_est = upperToFull(val, 0)
            # Giacomo 6/07. The following line's result were unused.
            # u, _ = np.linalg.eig(S_est)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            cov_out = np.linalg.inv(S_est)
            log_det[cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[cluster] = cov_out
            inverse_covariance[cluster] = S_est

        return {
            "log_det": log_det,
            "computed_covariance": computed_covariance,
            "inverse_covariance": inverse_covariance,
        }

    def train_clusters(self, complete_D_train, n, pool, train_clusters_arr):
        """
        Arguments:
        ----------
         - cluster_mean_info : dict
            Dictionary holds output values about the cluster means
         - cluster_mean_stacked_info : dict
            Dictionary holds output values about the cluster means
         - complete_D_train : 2D array
            Data array
         - empirical_covariances : dict
            Dictionary holds output covariances for each cluster
         - n : int
            Number of features in the data (not taking into accound the window)
         - pool : multiprocessing.Pool
         - train_clusters_arr : dict
            Contains the initial assignment of points to clusters.
        Returns:
         - optRes : list
            Contains new cluster assignments???
        """
        stacked_mean = {}
        end_mean = {}
        empirical_covariances = {}
        rho = 1
        admm_max_iters = 1000
        admm_eps_abs = 1e-6
        admm_eps_rel = 1e-6
        admm_verbose = False

        nw = n * self.window_size
        opt_res = [None for i in range(self.number_of_clusters)]
        for cluster in range(self.number_of_clusters):
            cluster_length = len(train_clusters_arr[cluster])
            if cluster_length != 0:
                D_train = complete_D_train[train_clusters_arr[cluster],:] # Nc x n*w
                D_mean = np.mean(D_train, axis=0) # n*w

                stacked_mean[cluster] = D_mean
                end_mean[cluster] = D_mean[(self.window_size - 1) * n:].reshape([1, n])
                ## Fit a model - OPTIMIZATION
                # Lambda is the same everywhere!
                lamb = np.zeros((nw, nw)) + self.lambda_parameter
                S = np.cov(np.transpose(D_train)) # n*w x n*w
                empirical_covariances[cluster] = S

                solver = ADMMSolver(lamb, self.window_size, n, rho, S)
                opt_res[cluster] = pool.apply_async(solver,
                    (admm_max_iters, admm_eps_abs, admm_eps_rel, admm_verbose))
        return {
            "stacked_mean": stacked_mean,
            "end_mean": end_mean,
            "empirical_covariances": empirical_covariances,
            "optimized_results": opt_res,
        }

    def stack_training_data(self, data, n, num_train_points, training_indices):
        """ Create the main data array by stacking the time-windows at each sample.
        Arguments:
         - data : 2D array
         - n : int
            Number of features in the input data
         - num_train_points : int
            Number of samples in the input data
         - training_indices : list[int]
            Indices of training samples in input data.
        """
        full_stacked = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            stack_indices = [training_indices[i + k]
                for k in range(self.window_size) if (i+k < num_train_points)]
            stack_data = data[stack_indices] # w x n
            if stack_data.shape[0] < self.window_size:
                stack_data = np.pad(stack_data,
                    [(0, self.window_size - stack_data.shape[0]), (0, 0)],
                    mode="constant")
            full_stacked[i] = stack_data.reshape(-1)

        return full_stacked

    def prepare_out_directory(self):
        out_path = ("%slambda_sparse=%.2fmax_clusters=%d" %
            (self.prefix_string, self.lambda_parameter, self.number_of_clusters + 1))
        mkdir_p(out_path)

        return out_path

    def load_data(self, input_ts):
        if isinstance(input_ts, str):
            data = np.loadtxt(input_ts, delimiter=",")
            print("%s Loaded data from file %s." % (time_str(), input_ts))
        elif isinstance(input_ts, np.ndarray):
            data = input_ts
            assert data.ndim == 2, "Data must have 2 dimensions."

        (m, n) = data.shape  # m: num of observations, n: size of observation vector
        return data, m, n

    def log_parameters(self):
        print("%s Starting TICC solver with params: " % (time_str()))
        print("\tlambda_sparse", self.lambda_parameter)
        print("\tswitch_penalty", self.switch_penalty)
        print("\tnum_cluster", self.number_of_clusters)
        print("\tnum stacked", self.window_size)

    def predict_clusters(self, model, train_data=None, test_data=None):
        """
        Given the current trained model, predict clusters.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        """
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            assert train_data is not None
            test_data = train_data

        print("%s Smoothing clusters (calculate log-likelihood)." % (time_str()))
        lle_all_points_clusters = self.smoothen_clusters(model, test_data,
            self.num_features)

        print("%s Updating cluster assignments" % (time_str()))
        # Clustered points: list of cluster indices, one per time-point
        clustered_points = updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)

        return clustered_points
