/*
This create contains a simple implementation of the 
[k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering).
*/

extern crate rand;
extern crate num;
#[macro_use(s)]
extern crate ndarray;

use std::ops::{Add, Sub, Mul, Div, Index};
use std::cmp::{PartialOrd, PartialEq};
use std::fmt::Debug;
use std::iter::FromIterator;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Ix, Axis, ScalarOperand};
use rand::Rng;
use rand::distributions::{Weighted, WeightedChoice, Sample};
use num::{cast, NumCast, Zero, Signed};

/*
Numeric value trait, defines the types that can be used for the value of each dimension in a
data point.
*/
pub trait Value: ScalarOperand + Add + Zero + Signed + NumCast + PartialOrd + Copy + Debug {}
impl<T> Value for T where T: ScalarOperand + Add + Zero + Signed + NumCast + PartialOrd + Copy + Debug {}

/*
Find the distance between two data points, given as Array rows.
*/
fn distance_squared<V: Value>(point_a: &ArrayView1<V>, point_b: &ArrayView1<V>) -> V {
    point_a.iter().zip(point_b.iter()).fold(num::Zero::zero(), |acc, v| {
        let delta = *v.0 - *v.1;
        return acc + (delta * delta);
    })
}

/*
Find the shortest distance between each data point and any of a set of mean points.
*/
fn closest_distance<V: Value>(means: &ArrayView2<V>, data: &ArrayView2<V>, k: u32) -> Vec<V> {
    let mut distances = Vec::with_capacity(k as usize);
    for d in data.outer_iter() {
        let mut closest = distance_squared(&d, &means.outer_iter().next().unwrap());
        for m in means.outer_iter() {
            let distance = distance_squared(&d, &m);
            if distance < closest {
                closest = distance;
            }
        }
        distances.push(closest);
    }
    return distances;
}

/*
This is a mean initialization method based on the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
initialization algorithm.
*/
fn initialize_plusplus<V: Value>(data: &ArrayView2<V>, k: u32) -> Array2<V> {
    assert!(k > 0);
    let mut means = Array2::zeros((k as usize, data.shape()[1]));
    let mut rng = rand::thread_rng();
    let data_len = data.shape()[0];
    let chosen = rng.gen_range(0, data_len) as isize;
    means.slice_mut(s![0..1, ..]).assign(&data.slice(s![chosen..(chosen + 1), ..]));
    for i in 1..k as isize {
		// Calculate the distance to the closest mean for each data point
        let distances = closest_distance(&means.slice(s![0..i, ..]).view(), &data.view(), k);
        // Find the largest distance to normalize the weights against
        let max_distance = distances.iter().fold(distances.iter().next().unwrap(), |mx, d|{
            if d > mx {
                d
            } else {
                mx
            }
        });
		// Pick a random point weighted by the distance from existing means
        let mut weights: Vec<Weighted<usize>> = distances.iter().zip(0..data_len).map(|d|{
            Weighted{weight: (std::u32::MAX / num::cast::<V, u32>(*max_distance).unwrap()) * num::cast::<V, u32>(*d.0).unwrap(), item: d.1}
        }).collect();
        let mut chooser = WeightedChoice::new(&mut weights);
        let chosen = chooser.sample(&mut rng) as isize;
        means.slice_mut(s![i..(i + 1), ..]).assign(&data.slice(s![chosen..(chosen + 1), ..]));
    }
    means
}

/*
Find the closest mean to a particular data point.
*/
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

/*
Calculate the index of the mean each data point is closest to (euclidean distance).
*/
fn calculate_clusters<V: Value>(data: &ArrayView2<V>, means: &ArrayView2<V>) -> Vec<Ix> {
    data.outer_iter()
    .map(|point|{
        closest_mean(&point.view(), means)
    })
    .collect()
}

/*
Calculate means based on data points and their cluster assignments.
*/
fn calculate_means<V: Value>(data: &ArrayView2<V>, clusters: &Vec<Ix>, old_means: &ArrayView2<V>, k: usize) -> Array2<V> {
    // TODO: replace old_means parameter with just its dimension, or eliminate it completely; that's all we need
    let (means, _) = clusters.iter()
        .zip(data.outer_iter())
        .fold((Array2::zeros(old_means.dim()), vec![1; k]), |mut cumulative_means, point|{
            {
                let total = cumulative_means.0.subview_mut(Axis(0), *point.0);
                let partial_mean = &point.1 * V::from(cumulative_means.1[*point.0]).unwrap();
                total + &partial_mean;
            }
            cumulative_means.1[*point.0] += 1;
            cumulative_means
        });
    means
}

// fn kmeans_lloyd<V: Value>(data: &ArrayView2<V>, k: usize)

#[cfg(test)]
mod tests {
    // extern crate csv;
    // use super::kmeans;
    
    #[test]
    fn test_distance() {
        use ndarray::arr1;
        use super::distance_squared;
        let a = arr1(&[1.0f32, 1.0f32]);
        let b = arr1(&[2.0f32, 2.0f32]);
        let c = arr1(&[1200.0f32, 1200.0f32]);
        let d = arr1(&[1, 1]);
        let e = arr1(&[1200, 1200]);
        assert_eq!(distance_squared(&a.view(), &b.view()), 2.0f32);
        assert_eq!(distance_squared(&a.view(), &c.view()), 2875202.0f32);
        assert_eq!(distance_squared(&d.view(), &e.view()), 2875202);
    }

    #[test]
    fn test_closest_distances() {
        use ndarray::arr2;
        use super::closest_distance;
        let a = arr2(&[
            [1, 1],
            [2, 2],
            [100, 4],
            [3, 100],
            [7, 88],
            [70, 20],
            [22, 12],
        ]);

        let m = arr2(&[
            [0, 0],
            [100, 0],
            [0, 100],
        ]);
        assert_eq!(closest_distance(&m.view(), &a.view(), m.len() as u32), vec![2, 8, 16, 9, 193, 1300, 628]);
    }

    #[test]
    fn test_closest_mean() {
        use ndarray::{arr1, arr2};
        use super::closest_mean;
        {
            let p = arr1(&[2, -1]);
            let m = arr2(&[
                [1, 1],
                [5, 100],
                [44, 65],
                [-5,-6]
            ]);
            assert_eq!(closest_mean(&p.view(), &m.view()), 0);
        }

        {
            let p = arr1(&[1024, 768]);
            let m = arr2(&[
                [1, 1],
                [5, 100],
                [512, 768],
                [-5,-6]
            ]);
            assert_eq!(closest_mean(&p.view(), &m.view()), 2);
        }
    }

    #[test]
    fn test_calculate_means() {
        use ndarray::{arr1, arr2};
        use super::calculate_means;
        {
            let d = arr2(&[
                [1, 1],
                [5, 100],
                [512, 768],
                [-5,-6]
            ]); 
            let c = vec![0, 1, 2, 3];
            let m = arr2(&[
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]
            ]);
            let means = calculate_means(&d.view(), &c, &m.view(), 4);
            println!("{:?}", means);
            assert!(false);
        }
    }
    
    // fn read_test_data() -> Vec<[f32; 2]> {
    //     let mut data_reader = csv::Reader::from_file("data/iris.data").unwrap();
    //     let mut data: Vec<[f32; 2]> = Vec::new();
    //     for record in data_reader.decode() {
    //         let (sl, _, pl, _, _): (f32, f32, f32, f32, String) = record.unwrap();
    //         data.push([sl, pl]);
    //     }
    //     println!("Read external data:");
    //     println!("{:?}", data);
    //     return data;
    // }
    // 
    // /*
    // Test the kmeans method with a basic f32 dataset loaded from a CSV file (in this case iris.data).
    // 
    // NOTE: This test just checks that the algorithm runs successfully; I haven't figured out a way
    // to check the validity of the data yet as this dataset has more than one local minima for this
    // algorithm. That doesn't mean one result is wrong, it's just the way the algorithm works.
    // */
    // #[test]
    // fn test_kmeans() {
    //     let data = read_test_data();
    //     let means = kmeans(&data[..], 3);
    //     println!("Got means {:?}", means);
    // }
    // 
    // /*
    // Test that the algorithm panics when k < 2 is given.
    // */
    // #[test]
    // #[should_panic(expected = "assertion failed")]
    // fn test_min_k() {
    //     let data = read_test_data();
    //     let means = kmeans(&data[..], 1);
    //     println!("Got means {:?}", means);
    //     // Should panic at this point
    //     println!("test_min_k failed, should have panicked");
    // }
    // 
    // /*
    // Test that the algorithm panics when no data is provided.
    // TODO: improve this behaviour?
    // */
    // #[test]
    // #[should_panic(expected = "assertion failed")]
    // fn test_min_data() {
    //     let data: Vec<[f32; 2]> = Vec::new();
    //     let means = kmeans(&data, 3);
    //     println!("Got means {:?}", means);
    //     // Should panic at this point
    //     println!("test_min_data failed, should have panicked");
    // }
}
