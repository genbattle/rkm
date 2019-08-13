/*
This crate contains a simple implementation of the
[k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering).
*/
extern crate num;
extern crate rand;
#[cfg(feature = "parallel")]
extern crate rayon;
#[macro_use(s)]
extern crate ndarray;
#[cfg(feature = "parallel")]
extern crate ndarray_parallel;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis, Ix, ScalarOperand};
#[cfg(feature = "parallel")]
use ndarray_parallel::prelude::*;
use num::{Float, NumCast, Zero};
use rand::distributions::{Distribution, Weighted, WeightedChoice};
use rand::prelude::*;
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::marker::Sync;
use std::ops::Add;

/// Numeric value trait, defines the types that can be used for the value of each dimension in a
/// data point.
pub trait Value:
    ScalarOperand + Add + Zero + Float + NumCast + PartialOrd + Copy + Debug + Sync + Send
{
}
impl<T> Value for T where
    T: ScalarOperand + Add + Zero + Float + NumCast + PartialOrd + Copy + Debug + Sync + Send
{
}

pub type RandomSeed = <rand::XorShiftRng as rand::SeedableRng>::Seed;

/// Algorithm configuration parameters, passed to `rkm::kmeans_lloyd` to specify:
/// * Random number generator seed
/// * The maximum number of iterations to terminate the algorithm at
/// * The minimum delta for all means from iteration to iteration
/// The algorithm will terminate if the maximum number of iterations is exceeded, or none of the means
/// change by at least the minimum delta distance, or the algorithm converges.
#[derive(Debug)]
pub struct Config<V: Value> {
    random_seed: Option<RandomSeed>,
    max_iterations: Option<u64>,
    min_delta: Option<V>,
}

impl<V: Value> Config<V> {
    /// Create a new config struct from a partial or complete set of parameters
    pub fn from(random_seed: Option<RandomSeed>, max_iterations: Option<u64>, min_delta: Option<V>) -> Config<V> {
        Config {
            random_seed: random_seed,
            max_iterations: max_iterations,
            min_delta: min_delta,
        }
    }

    /// Create an empty config struct
    pub fn empty() -> Config<V> {
        Config {
            random_seed: None,
            max_iterations: None,
            min_delta: None,
        }
    }
}

/// Find the distance between two data points, given as Array rows.
fn distance_squared<V: Value>(point_a: &ArrayView1<V>, point_b: &ArrayView1<V>) -> V {
    let mut distance = V::zero();
    for i in 0..point_a.shape()[0] {
        let delta = point_a[i] - point_b[i];
        distance = distance + (delta * delta)
    }
    return distance;
}

/// Find the shortest distance between each data point and any of a set of mean points (parallel version).
#[cfg(feature = "parallel")]
fn closest_distance<V: Value>(means: &ArrayView2<V>, data: &ArrayView2<V>) -> Vec<V> {
    data.outer_iter()
        .into_par_iter()
        .map(|d| {
            let mut iter = means.outer_iter();
            let mut closest = distance_squared(&d, &iter.next().unwrap());
            for m in iter {
                let distance = distance_squared(&d, &m);
                if distance < closest {
                    closest = distance;
                }
            }
            closest
        })
        .collect()
}

/// Find the shortest distance between each data point and any of a set of mean points.
#[cfg(not(feature = "parallel"))]
fn closest_distance<V: Value>(means: &ArrayView2<V>, data: &ArrayView2<V>) -> Vec<V> {
    data.outer_iter()
        .map(|d| {
            let mut iter = means.outer_iter();
            let mut closest = distance_squared(&d, &iter.next().unwrap());
            for m in iter {
                let distance = distance_squared(&d, &m);
                if distance < closest {
                    closest = distance;
                }
            }
            closest
        })
        .collect()
}

/// This is a mean initialization method based on the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
/// initialization algorithm (parallel version).
#[cfg(feature = "parallel")]
fn initialize_plusplus<V: Value>(data: &ArrayView2<V>, k: usize, seed: Option<RandomSeed>) -> Array2<V> {
    assert!(k > 1);
    assert!(data.dim().0 > 0);
    let mut means = Array2::zeros((k as usize, data.shape()[1]));
    let mut rng = match seed {
        Some(seed) => XorShiftRng::from_seed(seed),
        None => XorShiftRng::from_rng(rand::thread_rng()).unwrap(),
    };
    let data_len = data.shape()[0];
    let chosen = rng.gen_range(0, data_len) as isize;
    means
        .slice_mut(s![0..1, ..])
        .assign(&data.slice(s![chosen..(chosen + 1), ..]));
    for i in 1..k as isize {
        // Calculate the distance to the closest mean for each data point
        let distances = closest_distance(&means.slice(s![0..i, ..]).view(), &data.view());
        // Pick a random point weighted by the distance from existing means
        let distance_sum: f64 = distances
            .iter()
            .fold(0.0f64, |sum, d| sum + num::cast::<V, f64>(*d).unwrap());
        let mut weights: Vec<Weighted<usize>> = distances
            .par_iter()
            .zip(0..data_len)
            .map(|p| Weighted {
                weight: ((num::cast::<V, f64>(*p.0).unwrap() / distance_sum)
                    * ((std::u32::MAX) as f64))
                    .floor() as u32,
                item: p.1,
            })
            .collect();
        let mut chooser = WeightedChoice::new(&mut weights);
        let chosen = chooser.sample(&mut rng) as isize;
        means
            .slice_mut(s![i..(i + 1), ..])
            .assign(&data.slice(s![chosen..(chosen + 1), ..]));
    }
    means
}

/// This is a mean initialization method based on the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
/// initialization algorithm.
#[cfg(not(feature = "parallel"))]
fn initialize_plusplus<V: Value>(data: &ArrayView2<V>, k: usize, seed: Option<RandomSeed>) -> Array2<V> {
    assert!(k > 1);
    assert!(data.dim().0 > 0);
    let mut means = Array2::zeros((k as usize, data.shape()[1]));
    let mut rng = match seed {
        Some(seed) => SmallRng::from_seed(seed),
        None => SmallRng::from_rng(rand::thread_rng()).unwrap(),
    };
    let data_len = data.shape()[0];
    let chosen = rng.gen_range(0, data_len) as isize;
    means
        .slice_mut(s![0..1, ..])
        .assign(&data.slice(s![chosen..(chosen + 1), ..]));
    for i in 1..k as isize {
        // Calculate the distance to the closest mean for each data point
        let distances = closest_distance(&means.slice(s![0..i, ..]).view(), &data.view());
        // Pick a random point weighted by the distance from existing means
        let distance_sum: f64 = distances
            .iter()
            .fold(0.0f64, |sum, d| sum + num::cast::<V, f64>(*d).unwrap());
        let mut weights: Vec<Weighted<usize>> = distances
            .iter()
            .zip(0..data_len)
            .map(|p| Weighted {
                weight: ((num::cast::<V, f64>(*p.0).unwrap() / distance_sum)
                    * ((std::u32::MAX) as f64))
                    .floor() as u32,
                item: p.1,
            })
            .collect();
        let chooser = WeightedChoice::new(&mut weights);
        let chosen = chooser.sample(&mut rng) as isize;
        means
            .slice_mut(s![i..(i + 1), ..])
            .assign(&data.slice(s![chosen..(chosen + 1), ..]));
    }
    means
}

/// Find the closest mean to a particular data point.
fn closest_mean<V: Value>(point: &ArrayView1<V>, means: &ArrayView2<V>) -> Ix {
    assert!(means.dim().0 > 0);
    let mut iter = means.outer_iter().enumerate();
    if let Some(compare) = iter.next() {
        let mut index = compare.0;
        let mut shortest_distance = distance_squared(point, &compare.1);
        for compare in iter {
            let distance = distance_squared(point, &compare.1);
            if distance < shortest_distance {
                shortest_distance = distance;
                index = compare.0;
            }
        }
        return index;
    }
    return 0; // Should never hit this due to the assertion of the precondition
}

/// Calculate the index of the mean each data point is closest to (euclidean distance) (parallel version).
#[cfg(feature = "parallel")]
fn calculate_clusters<V: Value>(data: &ArrayView2<V>, means: &ArrayView2<V>) -> Vec<Ix> {
    data.outer_iter()
        .into_par_iter()
        .map(|point| closest_mean(&point.view(), means))
        .collect()
}

/// Calculate the index of the mean each data point is closest to (euclidean distance).
#[cfg(not(feature = "parallel"))]
fn calculate_clusters<V: Value>(data: &ArrayView2<V>, means: &ArrayView2<V>) -> Vec<Ix> {
    data.outer_iter()
        .map(|point| closest_mean(&point.view(), means))
        .collect()
}

/// Calculate means based on data points and their cluster assignments (parallel version)
#[cfg(feature = "parallel")]
fn calculate_means<V: Value>(
    data: &ArrayView2<V>,
    clusters: &Vec<Ix>,
    old_means: &ArrayView2<V>,
    k: usize,
) -> Array2<V> {
    // TODO: replace old_means parameter with just its dimension, or eliminate it completely; that's all we need
    let (mut means, counts) = clusters
        .par_iter()
        .zip(data.outer_iter().into_par_iter())
        .fold(
            || (Array2::zeros(old_means.dim()), vec![0; k]),
            |mut totals, point| {
                {
                    let mut sum = totals.0.subview_mut(Axis(0), *point.0);
                    let new_sum = &sum + &point.1;
                    sum.assign(&new_sum);
                    // TODO: file a bug about how + and += work with ndarray
                }
                totals.1[*point.0] += 1;
                totals
            },
        )
        .reduce(
            || (Array2::zeros(old_means.dim()), vec![0; k]),
            |new_means, subtotal| {
                let total = new_means.0 + subtotal.0;
                let count = new_means
                    .1
                    .iter()
                    .zip(subtotal.1.iter())
                    .map(|counts| counts.0 + counts.1)
                    .collect();
                (total, count)
            },
        );
    for i in 0..k {
        let mut sum = means.subview_mut(Axis(0), i);
        let new_mean = &sum / V::from(counts[i]).unwrap();
        sum.assign(&new_mean);
    }
    means
}

/// Calculate means based on data points and their cluster assignments.
#[cfg(not(feature = "parallel"))]
fn calculate_means<V: Value>(
    data: &ArrayView2<V>,
    clusters: &Vec<Ix>,
    old_means: &ArrayView2<V>,
    k: usize,
) -> Array2<V> {
    // TODO: replace old_means parameter with just its dimension, or eliminate it completely; that's all we need
    let (means, _) = clusters.iter().zip(data.outer_iter()).fold(
        (Array2::zeros(old_means.dim()), vec![0; k]),
        |mut cumulative_means, point| {
            {
                let mut mean = cumulative_means.0.subview_mut(Axis(0), *point.0);
                let n = V::from(cumulative_means.1[*point.0]).unwrap();
                let step1 = &mean * n;
                let step2 = &step1 + &point.1;
                let step3 = &step2 / (n + V::one());
                mean.assign(&step3);
                // TODO: file a bug about how + and += work with ndarray
            }
            cumulative_means.1[*point.0] += 1;
            cumulative_means
        },
    );
    means
}

/// Calculate means and cluster assignments for the given data and number of clusters (k).
/// Returns a tuple containing the means (as a 2D ndarray) and a `Vec` of indices that
/// map into the means ndarray and correspond elementwise to each input data point to give
/// the cluster assignments for each data point.
pub fn kmeans_lloyd<V: Value>(
    data: &ArrayView2<V>,
    k: usize,
    config: &Config<V>,
) -> (Array2<V>, Vec<usize>) {
    assert!(k > 1);
    assert!(data.dim().0 >= k);

    let mut old_means = initialize_plusplus(data, k, config.random_seed);
    let mut clusters = calculate_clusters(data, &old_means.view());
    let mut means = calculate_means(data, &clusters, &old_means.view(), k);

    while means != old_means {
        clusters = calculate_clusters(data, &means.view());
        old_means = means;
        means = calculate_means(data, &clusters, &old_means.view(), k);
    }

    (means, clusters)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_distance() {
        use super::distance_squared;
        use ndarray::arr1;
        let a = arr1(&[1.0f32, 1.0f32]);
        let b = arr1(&[2.0f32, 2.0f32]);
        let c = arr1(&[1200.0f32, 1200.0f32]);
        let d = arr1(&[1.0f32, 1.0f32]);
        let e = arr1(&[1200.0f32, 1200.0f32]);
        assert_eq!(distance_squared(&a.view(), &b.view()), 2.0f32);
        assert_eq!(distance_squared(&a.view(), &c.view()), 2875202.0f32);
        assert_eq!(distance_squared(&d.view(), &e.view()), 2875202.0f32);
    }

    #[test]
    fn test_closest_distances() {
        use super::closest_distance;
        use ndarray::arr2;
        let a = arr2(&[
            [1.0f32, 1.0f32],
            [2.0f32, 2.0f32],
            [100.0f32, 4.0f32],
            [3.0f32, 100.0f32],
            [7.0f32, 88.0f32],
            [70.0f32, 20.0f32],
            [22.0f32, 12.0f32],
        ]);

        let m = arr2(&[[0.0f32, 0.0f32], [100.0f32, 0.0f32], [0.0f32, 100.0f32]]);
        assert_eq!(
            closest_distance(&m.view(), &a.view()),
            vec![2.0f32, 8.0f32, 16.0f32, 9.0f32, 193.0f32, 1300.0f32, 628.0f32]
        );
    }

    #[test]
    fn test_closest_mean() {
        use super::closest_mean;
        use ndarray::{arr1, arr2};
        {
            let p = arr1(&[2.0f32, -1.0f32]);
            let m = arr2(&[
                [1.0f32, 1.0f32],
                [5.0f32, 100.0f32],
                [44.0f32, 65.0f32],
                [-5.0f32, -6.0f32],
            ]);
            assert_eq!(closest_mean(&p.view(), &m.view()), 0);
        }

        {
            let p = arr1(&[1024.0f32, 768.0f32]);
            let m = arr2(&[
                [1.0f32, 1.0f32],
                [5.0f32, 100.0f32],
                [512.0f32, 768.0f32],
                [-5.0f32, -6.0f32],
            ]);
            assert_eq!(closest_mean(&p.view(), &m.view()), 2);
        }
    }

    #[test]
    fn test_calculate_means() {
        use super::calculate_means;
        use ndarray::arr2;
        {
            let d = arr2(&[
                [0.0f32, 0.0f32],
                [2.0f32, 2.0f32],
                [4.0f32, 5.0f32],
                [5.0f32, 100.0f32],
                [128.0f32, 300.0f32],
                [512.0f32, 768.0f32],
                [-5.0f32, -6.0f32],
                [5.0f32, 6.0f32],
            ]);
            let c = vec![0, 0, 1, 1, 2, 2, 3, 3];
            let m = arr2(&[
                [0.0f32, 0.0f32],
                [0.0f32, 0.0f32],
                [0.0f32, 0.0f32],
                [0.0f32, 0.0f32],
            ]);
            let expected_means = arr2(&[
                [1.0f32, 1.0f32],
                [4.50f32, 52.5f32],
                [320.0f32, 534.0f32],
                [0.0f32, 0.0f32],
            ]);
            assert_eq!(calculate_means(&d.view(), &c, &m.view(), 4), expected_means);
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_min_k() {
        use super::kmeans_lloyd;
        use super::Config;
        use ndarray::arr2;
        {
            let d = arr2(&[[1.0f32, 1.0f32], [2.0f32, 2.0f32], [3.0f32, 3.0f32]]);
            kmeans_lloyd(&d.view(), 1, &Config::empty());
        }
    }

    #[test]
    fn test_small_kmeans() {
        use super::kmeans_lloyd;
        use super::Config;
        use ndarray::arr2;
        {
            let d = arr2(&[
                [1.0f32, 1.0f32],
                [2.0f32, 2.0f32],
                [3.0f32, 3.0f32],
                [1200.0f32, 1200.0f32],
                [956.0f32, 956.0f32],
                [1024.0f32, 1024.0f32],
                [1024.0f32, 17.0f32],
                [1171.0f32, 20.0f32],
            ]);
            let expected_means = arr2(&[
                [2.0f32, 2.0f32],
                [1097.5f32, 18.5f32],
                [1060.0f32, 1060.0f32],
            ]);
            let expected_clusters = vec![0, 0, 0, 2, 2, 2, 1, 1];
            let config = Config::from(Some((0 as u128).to_le_bytes()), None, None);
            let (means, clusters) = kmeans_lloyd(&d.view(), 3, &config);
            println!("{:?}", means);
            println!("{:?}", clusters);
            assert!(clusters == expected_clusters);
            assert!(means == expected_means);
        }
    }
}
