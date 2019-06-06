import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np
import warnings

from collections import defaultdict
from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state, gen_batches, check_array
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import effective_n_jobs


def plot2DLocations(channel_positions, neurons, locs, neuron_locs, ax=None, dims=[1,2], figsize=(8,8), alpha=.3, xlims=None, ylims=None, epochs=0, annotate=True):
    import colorsys
    
    N = 50
    HSV_tuples = [(x*1.0/N, .75, .75) for x in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
#     np.random.seed(0)
#     _color = 1.0 * np.random.permutation(len(set(neurons))) / len(set(neurons))
#     hs_c = plt.cm.hsv(_color[neurons])
#     RGB_tuples = hs_c
    if(ax == None):
        f, ax = plt.subplots(figsize=figsize)   
    if(ylims is not None):
        ax.set_ylim((ylims[0], ylims[1]))
    if(xlims is not None):
        ax.set_xlim((xlims[0], xlims[1]))
    ax.scatter(channel_positions[:,dims[0]], channel_positions[:,dims[1]], alpha=.3,  c="grey", marker="s", lw=0)

    for neuron in neurons:
        if(len(locs[neuron]) != 0):
            ax.scatter(np.asarray(locs[neuron])[:,dims[0]], np.asarray(locs[neuron])[:,dims[1]], c=RGB_tuples[neuron], alpha=alpha, s=8, rasterized=False, lw=0, edgecolor='')
            ax.scatter(neuron_locs[neuron][dims[0]], neuron_locs[neuron][dims[1]], s=80, marker='*', c="black", lw=0, edgecolor='');
        if(annotate):  
            text = ax.annotate(neuron, (neuron_locs[neuron][dims[0]] + 2, neuron_locs[neuron][dims[1]] + 2))
            text.set_fontsize(10)
#         text = ax.annotate("Epochs: " + str(epochs), (-33, 120))
#         text.set_fontsize(10)
    return ax

def plot3DLocations(channel_positions, neurons, locs, neuron_locs, ax=None, figsize=(8,8), loc_color='grey', alpha=.3, annotate=True):
    import colorsys
    N = 50
    np.random.seed(0)
    _color = 1.0 * np.random.permutation(len(set(neurons))) / len(set(neurons))
    hs_c = plt.cm.hsv(_color[neurons])
    RGB_tuples = hs_c
    if(ax == None):
        fig = plt.figure(figsize=(8,20))
        ax = fig.add_subplot(111, projection='3d')       
    ax.scatter(channel_positions[:,0], channel_positions[:,1], channel_positions[:,2], alpha=.7, c="black")

    for neuron in neurons:
        if(len(locs[neuron]) != 0):
            ax.scatter(np.absolute(np.asarray(locs[neuron])[:,0]), np.asarray(locs[neuron])[:,1], np.asarray(locs[neuron])[:,2], c=RGB_tuples[neuron], alpha=alpha)
            ax.scatter(neuron_locs[neuron][0], neuron_locs[neuron][1], neuron_locs[neuron][2], marker='*', c="black");
    return ax

def plotClusterResults(channel_positions, cluster_data, labels, neuron_locs, dims=[0,1], ax=None, figsize=(8,8), alpha=.3, clustering_alg='hdbscan', annotate=True, min_cluster_size=50):
    
    import colorsys
    if(clustering_alg == 'dbscan' or clustering_alg == 'hdbscan'):
        N = len(np.unique(labels)) - 1
    else:
        N = len(np.unique(labels))
    HSV_tuples = [(x*1.0/N, .75, .75) for x in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    if(ax == None):
        f, ax = plt.subplots(figsize=figsize)        
    ax.scatter(channel_positions[:,1], channel_positions[:,2], c="black", alpha=.5, marker="s")
    
    total_clusters_above_threshold = 0
    for i in range(N):
        inds = np.where(labels==i)
        if(inds[0].shape[0] >= min_cluster_size):
            total_clusters_above_threshold+=1
            ax.scatter(cluster_data[inds][:,dims[0]], cluster_data[inds][:,dims[1]], c=RGB_tuples[i], alpha=alpha)
            if(annotate):  
                text = ax.annotate(i, (np.mean(cluster_data[inds][:,dims[0]]), np.mean(cluster_data[inds][:,dims[1]])))
                text.set_fontsize(12)
                text.set_weight('bold')
    if(neuron_locs is not None):
        for neuron in range(50):  
            ax.scatter(neuron_locs[neuron][1], neuron_locs[neuron][2], s=100, marker='*', c="green");
    ax.set_title(str(clustering_alg) + ", Num clusters: "+ str(total_clusters_above_threshold))
    return ax

"""Mean shift clustering algorithm.
Mean shift clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These
candidates are then filtered in a post-processing stage to eliminate
near-duplicates to form the final set of centroids.
Seeding is performed using a binning technique for scalability.
"""

# Authors: Conrad Lee <conradlee@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Martino Sorbaro <martino.sorbaro@ed.ac.uk>



def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0,
                       n_jobs=None):
    """Estimate the bandwidth to use with the mean-shift algorithm.
    That this function takes time at least quadratic in n_samples. For large
    datasets, it's wise to set that parameter to a small value.
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input points.
    quantile : float, default 0.3
        should be between [0, 1]
        0.5 means that the median of all pairwise distances is used.
    n_samples : int, optional
        The number of samples to use. If not given, all samples are used.
    random_state : int, RandomState instance or None (default)
        The generator used to randomly select the samples from input points
        for bandwidth estimation. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Returns
    -------
    bandwidth : float
        The bandwidth parameter.
    """
    X = check_array(X)

    random_state = check_random_state(random_state)
    if n_samples is not None:
        idx = random_state.permutation(X.shape[0])[:n_samples]
        X = X[idx]
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:  # cannot fit NearestNeighbors with n_neighbors = 0
        n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            n_jobs=n_jobs)
    nbrs.fit(X)

    bandwidth = 0.
    for batch in gen_batches(len(X), 500):
        d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
        bandwidth += np.max(d, axis=1).sum()

    return bandwidth / X.shape[0]


# separate function for each seed's iterative loop
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    completed_iterations = 0
    while True:
        # Find mean of points within bandwidth
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth,
                                       return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break  # Depending on seeding strategy this condition may occur
        my_old_mean = my_mean  # save the old mean
        my_mean = np.mean(points_within, axis=0)
        # If converged or at max_iter, adds the cluster
        if (np.linalg.norm(my_mean - my_old_mean) < stop_thresh or
                completed_iterations == max_iter):
            return tuple(my_mean), len(points_within)
        completed_iterations += 1


def _mean_shift_multi_seeds(my_means, X, nbrs, max_iter):
    # Process a batch of seeds.
    res = []
    for my_mean in my_means:
        res.append(_mean_shift_single_seed(my_mean, X, nbrs, max_iter))
    return res


def mean_shift(X, bandwidth=None, seeds=None, bin_seeding=False,
               min_bin_freq=1, cluster_all=True, max_iter=300,
               n_jobs=None):
    """Perform mean shift clustering of data using a flat kernel.
    Read more in the :ref:`User Guide <mean_shift>`.
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input data.
    bandwidth : float, optional
        Kernel bandwidth.
        If bandwidth is not given, it is determined using a heuristic based on
        the median of all pairwise distances. This will take quadratic time in
        the number of samples. The sklearn.cluster.estimate_bandwidth function
        can be used to do this more efficiently.
    seeds : array-like, shape=[n_seeds, n_features] or None
        Point used as initial kernel locations. If None and bin_seeding=False,
        each data point is used as a seed. If None and bin_seeding=True,
        see bin_seeding.
    bin_seeding : boolean, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        Ignored if seeds argument is not None.
    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.
    cluster_all : booleazn, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.
    max_iter : int, default 300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 0.17
           Parallel Execution using *n_jobs*.
    Returns
    -------
    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.
    labels : array, shape=[n_samples]
        Cluster labels for each point.
    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_mean_shift.py
    <sphx_glr_auto_examples_cluster_plot_mean_shift.py>`.
    """

    if bandwidth is None:
        bandwidth = estimate_bandwidth(X, n_jobs=n_jobs)
    elif bandwidth <= 0:
        raise ValueError("bandwidth needs to be greater than zero or None,\
            got %f" % bandwidth)
    if seeds is None:
        if bin_seeding:
            seeds = get_bin_seeds(X, bandwidth, min_bin_freq)
        else:
            seeds = X
    n_samples, n_features = X.shape
    center_intensity_dict = {}

    # We use n_jobs=1 because this will be used in nested calls under
    # parallel calls to _mean_shift_single_seed so there is no need for
    # for further parallelism.
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)

    ncpus = effective_n_jobs(n_jobs)
    nseeds = int(len(seeds)//ncpus+1)
    print("number of seeds: "+str(len(seeds)))
    print("seeds/job: "+str(nseeds))
    print("using "+str(ncpus)+" cpus")

    # here each job gets its batch of seeds:
    all_res = Parallel(n_jobs=ncpus, max_nbytes=1e6, verbose=2)(
        delayed(_mean_shift_multi_seeds)
        (seeds[i*nseeds:(i+1)*nseeds], X, nbrs, max_iter) for i in range(ncpus))

    # retrieve results from batches
    for i in range(ncpus):
        for ii in range(len(all_res[i])):
            if all_res[i][ii] is not None:
                center_intensity_dict[all_res[i][ii][0]] = all_res[i][ii][1]

    if not center_intensity_dict:
        # nothing near seeds
        raise ValueError("No point was within bandwidth=%f of any seed."
                         " Try a different seeding strategy \
                         or increase the bandwidth."
                         % bandwidth)

    # POST PROCESSING: remove near duplicate points
    # If the distance between two kernels is less than the bandwidth,
    # then we have to remove one because it is a duplicate. Remove the
    # one with fewer points.
    sorted_by_intensity = sorted(center_intensity_dict.items(),
                                 key=lambda tup: (tup[1], tup[0]),
                                 reverse=True)
    sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
    unique = np.ones(len(sorted_centers), dtype=np.bool)
    nbrs = NearestNeighbors(radius=bandwidth,
                            n_jobs=1).fit(sorted_centers)
    for i, center in enumerate(sorted_centers):
        if unique[i]:
            neighbor_idxs = nbrs.radius_neighbors([center],
                                                  return_distance=False)[0]
            unique[neighbor_idxs] = 0
            unique[i] = 1  # leave the current point as unique
    cluster_centers = sorted_centers[unique]

    # ASSIGN LABELS: a point belongs to the cluster that it is closest to
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(cluster_centers)
    labels = np.zeros(n_samples, dtype=np.int)
    distances, idxs = nbrs.kneighbors(X)
    if cluster_all:
        labels = idxs.flatten()
    else:
        labels.fill(-1)
        bool_selector = distances.flatten() <= bandwidth
        labels[bool_selector] = idxs.flatten()[bool_selector]
    return cluster_centers, labels


def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """Finds seeds for mean_shift.
    Finds seeds by first binning data onto a grid whose lines are
    spaced bin_size apart, and then choosing those bins with at least
    min_bin_freq points.
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input points, the same points that will be used in mean_shift.
    bin_size : float
        Controls the coarseness of the binning. Smaller values lead
        to more seeding (which is computationally more expensive). If you're
        not sure how to set this, set it to the value of the bandwidth used
        in clustering.mean_shift.
    min_bin_freq : integer, optional
        Only bins with at least min_bin_freq will be selected as seeds.
        Raising this value decreases the number of seeds found, which
        makes mean_shift computationally cheaper.
    Returns
    -------
    bin_seeds : array-like, shape=[n_samples, n_features]
        Points used as initial kernel positions in clustering.mean_shift.
    """

    # Bin points
    bin_sizes = defaultdict(int)
    for point in X:
        binned_point = np.round(point / bin_size)
        bin_sizes[tuple(binned_point)] += 1

    # Select only those bins as seeds which have enough members
    bin_seeds = np.array([point for point, freq in six.iteritems(bin_sizes) if
                          freq >= min_bin_freq], dtype=np.float32)
    if len(bin_seeds) == len(X):
        warnings.warn("Binning data failed with provided bin_size=%f,"
                      " using data points as seeds." % bin_size)
        return X
    bin_seeds = bin_seeds * bin_size
    return bin_seeds


class MeanShift(BaseEstimator, ClusterMixin):
    """Mean shift clustering using a flat kernel.
    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.
    Seeding is performed using a binning technique for scalability.
    Read more in the :ref:`User Guide <mean_shift>`.
    Parameters
    ----------
    bandwidth : float, optional
        Bandwidth used in the RBF kernel.
        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).
    seeds : array, shape=[n_samples, n_features], optional
        Seeds used to initialize kernels. If not set,
        the seeds are calculated by clustering.get_bin_seeds
        with bandwidth as the grid size and default values for
        other parameters.
    bin_seeding : boolean, optional
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        default value: False
        Ignored if seeds argument is not None.
    min_bin_freq : int, optional
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds. If not defined, set to 1.
    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ :
        Labels of each point.
    Examples
    --------
    >>> from sklearn.cluster import MeanShift
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = MeanShift(bandwidth=2).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering.predict([[0, 0], [5, 5]])
    array([1, 0])
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    MeanShift(bandwidth=2, bin_seeding=False, cluster_all=True, min_bin_freq=1,
         n_jobs=None, seeds=None)
    Notes
    -----
    Scalability:
    Because this implementation uses a flat kernel and
    a Ball Tree to look up members of each kernel, the complexity will tend
    towards O(T*n*log(n)) in lower dimensions, with n the number of samples
    and T the number of points. In higher dimensions the complexity will
    tend towards O(T*n^2).
    Scalability can be boosted by using fewer seeds, for example by using
    a higher value of min_bin_freq in the get_bin_seeds function.
    Note that the estimate_bandwidth function is much less scalable than the
    mean shift algorithm and will be the bottleneck if it is used.
    References
    ----------
    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.
    """
    def __init__(self, bandwidth=None, seeds=None, bin_seeding=False,
                 min_bin_freq=1, cluster_all=True, n_jobs=None):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.min_bin_freq = min_bin_freq
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Perform clustering.
        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.
        y : Ignored
        """
        X = check_array(X)
        self.cluster_centers_, self.labels_ = \
            mean_shift(X, bandwidth=self.bandwidth, seeds=self.seeds,
                       min_bin_freq=self.min_bin_freq,
                       bin_seeding=self.bin_seeding,
                       cluster_all=self.cluster_all, n_jobs=self.n_jobs)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")
        print('predicting')

        return pairwise_distances_argmin(X, self.cluster_centers_)