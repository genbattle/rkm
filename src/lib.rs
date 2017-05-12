/*
This create contains a simple implementation of the 
[k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering).
*/

extern crate rand;
extern crate num;
extern crate ndarray;

use std::ops::{Add, Sub, Mul, Div, Index};
use std::cmp::{PartialOrd, PartialEq};
use std::fmt::Debug;
use std::iter::FromIterator;
use ndarray::{Array1, Array2, arr1, arr2};
use rand::Rng;
use rand::distributions::{Weighted, WeightedChoice};

/*
Numeric value trait, defines the types that can be used for the value of each dimension in a
data point.
*/
pub trait Value: num::Signed/* + From<usize>*/ + PartialOrd + Copy + Debug {}
impl<T> Value for T where T: num::Signed/*+ From<usize>*/ + PartialOrd + Copy + Debug {}

/*
Find the distance between two data points, given as Array rows.
*/
fn distance_squared<V: Value>(point_a: &Array1<V>, point_b: &Array1<V>) -> V {
    point_a.iter().zip(point_b.iter()).fold(num::Zero::zero(), |acc, v| {
        let delta = *v.0 - *v.1;
        return acc + (delta * delta);
    })
}

// /*
// Find the shortest distance between each data point and any of a set of mean points.
// TODO: This will be used for km++ initialization
// TODO: This will need to return rand::distributions::Weighted in order to work
// */
// fn closest_distance<V: Value>(means: &Array2<V>, data: &Array2<V>, k: u32) -> Vec<V> {
//     let mut distances = Vec::with_capacity(k as usize);
//     for i in 0..data.dim().0 {
//         let mut closest = distance_squared(&data.subview(0, i), &means.subview(0, 0));
//         for j in 0..means.dim().0 {
//             let distance = distance_squared(&data.subview(0, i), &means.subview(0, j));
//             if distance < closest {
//                 closest = distance;
//             }
//         }
//         distances.push(closest);
//     }
//     return distances;
// }

// // TODO: kmeans++ initialization
// /*
// This is a mean initialization method based on the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
// initialization algorithm.
// */
// fn initialize_plusplus<V: Value>(data: &Array2<V>, k: u32) -> Array2<V> {
//     assert!(k > 0);
//     let mut means = Array::<V, (Ix, Ix)>::zeros((k, data.dim().1));
//     let mut rng = rand::thread_rng();
//     let data_len = data.dim().0;
//     means.subview(0, 0).assign(&data.subview(0, rng.gen_range(0, data_len)));
//     for i in 1..data_len {
// 		// Calculate the distance to the closest mean for each data point
//         let distances = closest_distance(&means.slice(&[Si(0, Some(i as i32), 1), S]), data, k);
// 		// Pick a random point weighted by the distance from existing means
//         let weights = distances.iter().zip(0..data_len).map(|d|{
//             Weighted{weight: *d.0, item: d.1}
//         }).collect();
//     }
//     return means;
// }

// /*
// Find the closest mean to a particular data point.
// */
// fn closest_mean<V: Value>(point: &ArrayView1<V>, means: &ArrayView2<V>) -> Ix {
//     // TODO: Check dimensionality/size of means and point
//     let mut shortest_distance = distance_squared(point, &means.subview(0, 0));
//     let mut index: Ix = 0;
//     for i in 0..means.dim().0 {
//         let distance = distance_squared(point, &means.subview(0, i));
//         if distance < shortest_distance {
//             shortest_distance = distance;
//             index = i;
//         }
//     }
//     return index;
// }

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
        assert_eq!(distance_squared(&a, &b), 2.0f32);
        assert_eq!(distance_squared(&a, &c), 2875202.0f32);
        assert_eq!(distance_squared(&d, &e), 2875202);
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
