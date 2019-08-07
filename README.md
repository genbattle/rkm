# RKM - Rust k-means #

[![docs](https://docs.rs/rkm/badge.svg)](https://docs.rs/rkm/latest/rkm/) [![crates.io](https://img.shields.io/crates/v/rkm.svg)](https://crates.io/crates/rkm)

A simple [Rust](https://www.rust-lang.org) implementation of the [k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering) based on a C++ implementation, [dkm](https://github.com/genbattle/dkm).

This implementation is generic, and will accept any type that satisfies the trait requirements. At a minimum, numeric floating point types built into rust should be supported. Uses rayon for parallelism to improve scalability at the cost of some performance on small data sets.

The `parallel` feature enables parallelism to speed up the algorithm in complex cases. The parallel algorithm may be slower than the non-parallel algorithm for small data sets, but is much faster for data sets with high dimensionality. Make sure you benchmark your use case with both configurations before deciding which to use.

Known to compile against Rust stable 1.32.0.

### Usage ###

Calculate the k-means clusters for a set of data by calling `rkm::kmeans_lloyd` with your dataset in a 2D `ndarray` array and the number of clusters you would like to segment the data into. The return value will be a tuple containing the cluster means/centroids (as a 2D `ndarray`) and a `Vec` of indices that map each of the input data points to an element of the means array.

See `src/example.rs` for a simple usage example.

### TODOs ###
* Termination conditions (iterations and delta).

### Data ###
 A small set of benchmarks for this library is included in `src/bench.rs`. The data sets are as follows:

`iris.data.csv` natural data taken from measurements of different iris plants. 150 points, 2 dimensions, 3 clusters. Source: [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/Iris).

`s1.data.csv` synthetic data. 5000 points, 2 dimensions, 15 clusters. Source: P. Fränti and O. Virmajoki, "Iterative shrinking method for clustering problems", _Pattern Recognition_, 39 (5), 761-765, May 2006.

`birch3.data.csv` synthetic data large set. 100000 points, 2 dimensions, 100 clusters. Source: Zhang et al., "BIRCH: A new data clustering algorithm and its applications", _Data Mining and Knowledge Discovery_, 1 (2), 141-182, 1997

`dim128.data.csv` synthetic data with high dimensionality. 1024 points, 128 dimensions, 16 clusters. Source: P. Fränti, O. Virmajoki and V. Hautamäki, "Fast agglomerative clustering using a k-nearest neighbor graph", _IEEE Trans. on Pattern Analysis and Machine Intelligence_, 28 (11), 1875-1881, November 2006

Compared to [dkm](https://github.com/genbattle/dkm) this implementation is slower for the small iris and s1 data sets, but faster for the `dim128` and `birch3` data sets.

### Licensing ###
 This code is licensed under the MIT license.
