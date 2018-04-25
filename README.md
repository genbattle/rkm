# RKM - Rust k-means #

A simple [Rust](https://www.rust-lang.org) implementation of the [k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering) based on a C++ implementation, [dkm](https://github.com/genbattle/dkm).

This implementation is generic, and will accept any type that satisfies the trait requirements. At a minimum, numeric types including the integer and floating point types built into rust should be supported.

A small benchmark for this library is included in `src/bench.rs`. This benchmark yields a run time of 0.36ms per call to `rkm::kmeans_lloyd` on a 1.7GHz Intel i5-4210U processor. If you are looking for a more performant implementation of the k-means algorithm then you should look into alternatives like [dkm](https://github.com/genbattle/dkm).

Pull requests to improve performance or other aspects of this library are actively encouraged.

Known to compile against Rust stable 1.17.0.

### TODOs ###
* CI
* Documentation
* Optimization

### Data ###
The iris.data file contains data obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris). The data contains the following columns:
 - sepal length (cm)
 - sepal width (cm)
 - petal length (cm)
 - petal width (cm)
 - class
 
 This data is used as a small test/benchmark of the k-means algorithm.

### Licensing ###
 This code is licensed under the MIT license.
