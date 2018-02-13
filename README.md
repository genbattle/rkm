# RKM - Rust k-means #

A simple [Rust](https://www.rust-lang.org) implementation of the [k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering) based on a C++ implementation, [dkm](https://github.com/genbattle/dkm).

This implementation is generic, and will accept any type that satisfies the trait requirements. At a minimum, numeric types including the integer and floating point types built into rust should be supported.

Known to compile against Rust stable 1.17.0.

### TODOs ###
* CI
* More unit tests
* Documentation

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
