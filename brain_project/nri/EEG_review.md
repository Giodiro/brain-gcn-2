Definitions from [here][static connectivity review]
FC: functional connectivity - the temporal correlation (in terms of statistically significant dependence between distant brain regions) among the activity of different neural assemblies.
EC: effective connectivity - the direct or indirect influence that one neural system exerts over another. Describes dynamic directional interactions among brain regions.

Two different lines of research: one concerns static analysis of connectivity, reviewed [here][static connectivity review]; another concerns dynamic functional connectivity (dFC) and is reviewed [here][dynamic connectivity review]


### Static FC

The static FC analysis prominently features model based techniques such as DCM, data-driven causal techniques (Granger causality-based partial directed coherence (PDC)), linear and non-linear methods (correlation, coherence, synchronization methods), and information-based techniques.
Assumes stationarity within the whole time-period!


### Dynamic FC

The dynamic FC analysis is mostly based on sliding window estimates of Pearson correlation. However, this is not the only thing:
#### 1. Limitations of the sliding window
Choice of the window length $W$ means trading off specificity and sensitivity: short window lengths risk introducing spurious fluctuations (from noisy data), and have too few samples for reliably estimating correlation; long windows make it impossible to detect temporal variations which occur within the window itself.
Tapering is often used to discount observations closer to the window boundary which decreases sensitivity to outliers.
Dynamic connectivity detection (DCD): detect discrete connectivity changes, and then use the identified windows for correlation estimation. This is somewhat similar to the [HMM model][hmm spectral connectivity]
There are other approaches which go in the direction of using more refined models of covariance/correlation between time courses.

#### 2. Joint time-frequency analysis
However, more work is now going into analysing time-varying aspects of the frequency-derived networks as well.

#### 3. Measuring within-window connectivity
Most common is use of Pearson correlation. Often sparsity is imposed, either on the covariance matrix itself, or on the precision matrix. This is useful since there are only a limited number of time-points for the estimation.
Higher order analyses have been done, such as through ICA (set of components for each window); IVA (independent vector analysis); regional homogeneity (ReHo)

#### 4. Dynamic Graph Analysis
Once the dFC matrices have been obtained, their analysis can be done with time-varying graph measures. 2 metrics in particular: efficiency describes the ease with which a signal can travel between two brain regions, modularity quantifies the extent to which the network is organized into a set of compact communities.

#### 5. Extracting dFC States
Clustering the time-series into states. Inputs are typically taken as concatenated connectivity matrices across time and subjects.
K-means clustring quite common ([for example][sliding window application]).
Temporal ICA has been used to obtain states which are maximally mutually independent. Unlike clustering tICA states are not exclusive, and can overlap.

#### 6. Future Directions -- Single frame estimates
Single frame estimates
Temporal modeling
TODO: Fill this in. It's not very understandable from the review, probably need to read the references.



## Spectrally resolved fast transient brain states in electrophysiological data - [10.1016/j.neuroimage.2015.11.047][hmm spectral connectivity]
Combines multivariate autoregressive model (MAR) and HMMs, to give a complete (temporal, spatial, frequency) description of MEG states.
Here we only describe the methods, but the paper also contains a considerable amount of experiments both simulated and on real-world data.

Complex Bayesian model, on top of the HMM with MAR observation model:
Priors on the ARD coefficients, ...
Inference with variational Bayes, maximizing likelihood and obtaining the MAR coefficients (which on their own can be taken as estimate of connectivity).
However, since the MAR connectivity estimate strongly depends on the set of lags chosen (which in the paper are non-uniformly distributed, thus unsuitable), they re-estimate connectivity using a multi-taper approach (nonparametric).
Their *statewise multitaper* extends the multitaper by allowing to make different estimates based on the states predicted by the HMM. In particular, they weigh the estimate of connectivity (in the frequency domain) for a given state by the HMM state likelihood of any given point. They estimate:
 - Power spectral density (PSD)
 - coherence
 - Partial directed coherence (PDC), without using the autoregressive coefficients (Wilson algorithm something).

They deal with inherent sign ambiguity of correlation estimates with a simple heuristic: all signs must be the same (between the same two ROIs) in all trials.

Signal leakage:
spatial leakage is induced by stationary correlations between source reconstruction weights. But HMM-MAR is identifying states as they change over time, hence cannot be influenced by temporally stationary spatial leakage.
However, since the multitaper is used at the end, spatial leakage can be present -> test for significance in spectral features by looking for differences from the global time-averaged spectral features.

As a followup, the same authors also wrote a [paper][https://doi.org/10.1016/j.neuroimage.2017.06.077] addressing computational complexity of the method. This improvement comes from using stochastic variational inference, where the stochastic part comes from using a subset of subjects at a time.
While on the original paper they only had 2-dimensional time-series, here they use 50 dimensions (maximum across various experiments) for fMRI data; only 2 dimensions for MEG data though.



## Dynamics of large-scale electrophysiological networks: A technical review [10.1016/j.neuroimage.2017.10.003][https://doi.org/10.1016/j.neuroimage.2017.10.003]
Another review for dFC analysis, this time based on EEG signals instead of fMRI!
Discussion about signal leakage in the first place.

## Bayesian Structure Learning for Dynamic Brain Connectivity [http://proceedings.mlr.press/v84/andersen18a/andersen18a.pdf]

Model estimates covariances which vary smoothly over time, with an instantaneous decomposition into a collection of spatially sparse components.
Evaluation: 1) synthetic data; 2) qualitative evaluation on brain data; 3) classification on brain data.
Assumptions: Temporally smooth dynamics (continuous rather than discrete switching between states).

Model:
 - Data $\bm{x}_t^n$ is modelled as multivariate Gaussian with 0 mean and covariance $\bm{\Sigma}_t^n$.
 - Modeling of the covariance matrix: define $\callig{S}$ a dictionary of K DxD covariance matrix components. $\bm{\Sigma}_t^n$ is then a non-negative weighted sum of components of $\callig{S}$ with mixing weights $\alpha_{k,t}^n$ which control the dynamics of each subject. Also a noise term is added.
 - Neuroscience perspective: each element of $\callig{S}$ represents the connectivity matrix of a cognitive process. At any time there may be a superposition of such processes at work.
 - 2 priors on $\alpha$: sparsity (only few cognitive processes at any given time), temporal smoothness (the covariance changes slowly over time). Since the coefficients are sparse one only needs to specify an upper bound K, and the model will automaticaly adjust the number of used covariance components!
 - $\alpha ~ \callig{GP}(\bm{m}_k^n, \bm{C}_k^n)$ is modelled as a Gaussian process. Prior covariance can be used for enforcing temporal smoothness
 - each S matrix is sparse, symmetric, rank 1. Spike-and-slab prior is enforced on the rank-1 components of S to encourage sparsity (Bernoulli times normal distribution).
 - Learning is done with mean-field variational inference!

Evaluation:
 - mean of GP: constant; covariance: Matern + scaled identity.
 - Simulated data
    use the Log-Euclidean Riemannian Metric (LERM) to compute the distance between estimated covariance, and ground truth sequence.
    Experiments with K=4, 145 time points, dimensions between 10,50.
    Try on both continuous and discrete switching dynamics. The first is exactly equal to model specification with state mixing coefficients varying continuously; in the second states activate or deactivate discretely, thus the GP has a harder time estimating coefficients.
 - fMRI motor task data

## Random notes

Power and phase coupling connectivity (spectrally resolved): at each specific frequency band, the correlation between power (i.e. amplitude) and phase (i.e. whether the oscillations are synchronized). Maybe power is just referred to as the amplitude of individual nodes, instead of pairwise correlation?
These connectivity patterns evolve over time -> temporal resolution


Presentation on the HMM stuff: http://www.humanbrainmapping.org/files/2016/ED/GraphTheory_Woolrich_Mark_v2.pdf

Unsupervised clustering with LSTMs (Schmidhuber): ftp://ftp.idsia.ch/pub/juergen/icann2001unsup.pdf


### [Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks][https://www.nature.com/articles/s41467-018-05316-z]
Novel method for identifying large-scale phase-coupled network dynamics.

TDE-HMM Model (time-delay embedded HMM):
Improves on the MVAR observation model, since it can scale more easily to whole brain (larger number of regions, in evaluation they use 42 ROIs) (MVAR has squared number of parameters in the ROI count).
The HMM has a MV Gaussian observation model on a temporal embedding of the data: each data-point contains not just the instantaneous values, but also the values at lags 1 to L. So the Gaussian emission covariance at a given state models the same autocorrelation coefficients as the MVAR model. Observations describe neural activity over a time window. Since this would require estimation of a (#ROIs * # time-lags)^2 covariance matrix for each state, which is typically quite large, a PCA decomposition of each observation was used before the HMM. This also naturally allows the HMM to focus on slower frequencies in the data.

Additional preprocessing: they use source-space data, so must use an appropriate signal-leakage reduction step (from "A symmetric multivariate leakage correction for MEG connectomes"), which eliminates zero-lag on the whole time-series level, but not necessarily on the level of smaller windows.
For correcting dipole sign ambiguity they use a similar approach as previous Vidaurre paper, of finding the sign which is most coherent with the data.

Spectral information is extracted from the HMM using the statewise multitaper, to avoid incurring in any PCA induced bias. NMF is used to factorize the identified power and spectral coherence matrices into different components / frequency modes in a data-driven way. This roughly recovers the standard frequency bands.

Evaluation:
Some resting state MEGs with 42 states. Operates in source space.

Discussion:
Large-scale networks in resting-state MEGs can be well described by repeated visits to short-lived transient brain states. The time spent in each state is shorter than previously though (i.e. 50-100 ms), and often encompasses less than one full cycle for given frequencies.
2 high coherence, high power states are identified and are hypothesized to be a decomposition of the default mode network (DMN). One is frontal one is posterior, and they occur in different frequency bands. Discussion is very much based on Neuro. Further short discussion about EEG micro-states framework (which is however only based on power, and disregards phase).

Comments:
Obviously using lags on the raw time-series means this is equivalent to a sliding window method. However, this seems to be an inherent fact of life if one wants to estimate spectral content from raw time-series.


## Application of GNNs to EEGs:
 1. [EEG-based video identification using graph signal modeling and graph convolutional neural network][https://arxiv.org/pdf/1809.04229.pdf]
 2. [EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks][https://doi.org/10.1109/TAFFC.2018.2817622]
 3. [Brain2Object: Printing your mind from brain signals with spatial correlation embedding][https://arxiv.org/abs/1810.02223]
 4. [Deep neural networks on graph signals for brain imaging analysis][https://arxiv.org/abs/1705.04828v1]
Overall these papers are all useless. None of them estimates a dynamic graph, and even their static graph approaches are mostly quite bad, and unprincipled.

### EEG-based video identification using graph signal modeling and graph convolutional neural network
Not a very good paper.

Extract signals from different bands. Power and entropy of the bands are used as signals on graphs. Graphs are defined in the electrode space (no source reconstruction step. This is probably not a good approach.)
Graph creation (static graphs are used throughout):
 1. Create intra-band graphs using 1) correlation-based (Pearson correlation used as edge weights, connections are sparsified using top-k approach) 2) distance-based (inverse distance between the electrodes used as edge weights, also sparsified with top-k) 3) random method (Erdos-Renyi random graph with edge probability $p$).
 2. Merge the graphs: 1) merge them into a single graph without connections between components (block diagonal) 2) merge them into a single graph imposing connectivity between pairs of vertices corresponding to the same electrods.

Graph Neural Network: They use ChebNet (spectral GNN based on Chebyshev polynomials to parametrize the filters), with a coarsening step to reduce the number of vertices at each layer.

Experiments:
Training on EEGs for 32 subjects watching videos (DEAP). Try to classify which video they're watching. Train/test split completely disregards subject data, so not very representative results.
Their second best result is obtained on the random graph (63.19%), so the graph is pretty useless in this setting-.- Useless paper.


### EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks
The term dynamical in the title is misleading: the graphs are static. They learn a graph from the data (thus the inferred graph changes during training, as it converges to the solution).

Graph Neural Network: ChebNet is used (spectral GNN).
Model: Basically they find a single adjacency matrix which is used as feature for subsequent classification. Learning of the adjacency matrix is done by backpropagating the loss. They experiment with a variety of node features (extracted in different spectral bands), finding that differential entropy (DE) leads to best performance.

Experimental evaluation on various datasets for emotion prediction, more convincing than the other video identification paper (they also do an subject independent test), but still not very interpretable. The adjacency matrix resulting is most likely bs since it's not task dependent but is shared between all different tasks.


### Brain2Object: Printing your mind from brain signals with spatial correlation embedding
Not 100% related. 1) multi-class Common Spatial Pattern?? for extracting features from EEG 2) Dynamical Graph Representation of EEG signals to embed into a graph 3) CNN to aggregate the graph representations.

CSP: Average covariance matrix for the samples from every distinct class. Then attempt to transform the original data so as to maximize the distance between different classes.
DGR: transform it into a ?? fully connected graph ?? by aggregating 1-hop distance signals?? This is super weird, but the end result is again a matrix with a row for each electrode.
CNN: Slap a CNN on top to classify.

The experiments are super cool! They detect the object you're thinking about (I suppose it can hardly generalize to objects not seen during training) and uses a 3D printer to print it. Nice but the model makes little sense.


### Deep neural networks on graph signals for brain imaging analysis

Again use ChebNet as a GNN.

Estimate a graph using granger causality connectivity. Then on top of the graph, with (unknown) features, stack ChebNet to produce transformed node features. Finally add an autoencoder on top which tries to reconstruct the original (unkown) features. Training is done via MSE loss. It's not clear what the purpose of the autoencoder would be, as there is no comparison *without* it. Experiments show minor improvement over using AE without the GNN.


##Not with graphs. Deep learning and EEGs:
 1. [Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks][https://arxiv.org/pdf/1511.06448.pdf] (ICLR16)
 2. [Real-Time sleep staging using deep learning on a smartphone for a wearable EEG][https://arxiv.org/pdf/1811.10111v2.pdf]
 3. [Automated Classification of Sleep Stages and EEG Artifacts in Mice with Deep Learning][https://arxiv.org/pdf/1809.08443v1.pdf]
 4. [A large-scale evaluation framework for EEG deep learning architectures][https://arxiv.org/pdf/1806.07741v2.pdf]
 5. [Multitaper Spectral Estimation HDP-HMMs for EEG Sleep Inference][https://arxiv.org/pdf/1805.07300v1.pdf]
 6. [Deep CNNs for interpretable analysis of EEG sleep stage scoring][https://arxiv.org/pdf/1710.00633v1.pdf]

### Multitaper Spectral Estimation HDP-HMMs for EEG Sleep Inference

wide-sense stationary (WSS) stochastic process: mean and autocovariance of the stochastic process is invariant wrt time.

Divide data in windows of size J. Transform the time-series in each window using a discrete fourier transform for J/2 different frequencies. Lemma: if y_t is a WSS time series, the DFT real and imaginary coefficients are normally distributed (i.e. are independent both of each other within the same frequency and between frequencies) as J->infinity. The distribution depends on the true power spectral density at a given frequency.

Multitaper Spectral Estimation:
 Problem of estimating the true PSD $f(w_j)$ from the DFT coefficients:
   1. Periodgram: $\hat{f}(w_j) = ||y_t^{(F)}(w_j)||^2$: the dot product of the coefficients. Has high variance and bias.
   2. Multitaper: Optimizes reduction in bias/variance by applyinc specific tapers (windowing functions) to the observed time-series, in order to obtain M (one for each taper) pseudo-observations (similar to the DFT coefficients, but tapered!). Then the final PSD estimate is the mean of the periodgrams for each tapered coefficient.

Generative Model:
Follows HDP-HMM framework. The observation model generates DFT coefficients for each time window $t$, frequency band of interest $w_j$ and taper $m$.
The likelihood is based on the lemma, where the true PSD (std of the normal distribution) depends on the state. The likelihood of DFT coefficients is a multivariate normal with mean 0 and diagonal covariance. The diagonal elements are the true PSDs for given frequency and state.
These true PSDs have an inverse gamma prior. a dirichlet process decides on the state proportions and transition distributions. Truncated GEM for stick breaking.

Inference:
Beam sampling to sample whole trajectories from model posterior (necessary due to the infinite dimensions of the DP). Inference is done separately for each subject, so there a second clustering step is necessary to merge states from different subjects. Clustering is done based on the MAP estimates of state-specific PSD, and the Gaussian distribution which it induces. Distance is the symmetric KL divergence between the 2 distributions.

Evaluation:
Simulated sleep data. Time-series generated using different spectral characteristics for each discrete state. The states all have different oscillatory components at different frequencies (data generated using a "oscillation components time series decomposition method").
Sleep data. Single channel used. Overall good evaluation, shows different states, good classification accuracy (Spearman rank is used as a metric), existence of different sub-states within major states.
Since a single channel is used (is this fundamental limitation, or a complexity limitation due to use of MCMC for inference?), there is no analysis concerning brain regions - functional connectivity.



## Random notes

Two ways of seeing the problem:
  1. estimating covariance matrix
  2. estimating the graph

Another decision:
  1. Hard clustering (i.e. each time-point is characterized by a single true state, like in HMMs)
  2. Soft clustering: time-points consist of a superposition of many different cognitive processes / states.



[static connectivity review]: https://doi.org/10.1016/j.compbiomed.2011.06.020
[dynamic connectivity review]: https://doi.org/10.1016/j.neuroimage.2016.12.061
[hmm spectral connectivity]: https://doi.org/10.1016/j.neuroimage.2015.11.047
[sliding window application]: https://doi.org/10.1038/s41467-018-03462-y
