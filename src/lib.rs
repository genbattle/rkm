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
use ndarray::{Array, Dimension, Ix};

/*
Numeric value trait, defines the types that can be used for the value of each dimension in a
data point.
*/
pub trait Value: num::Signed/* + From<usize>*/ + PartialOrd + Copy + Debug {}
impl<T> Value for T where T: num::Signed/*+ From<usize>*/ + PartialOrd + Copy + Debug {}

fn distance_squared<V: Value>(point_a: &Array<V, Ix>, point_b: &Array<V, Ix>) -> V {
    point_a.iter().zip(point_b.iter()).fold(num::Zero::zero(), |acc, v| {
        let delta = *v.0 - *v.1;
        return acc + (delta * delta);
    })
}

fn closest_distance<V: Value>(means: &Array<V, (Ix, Ix)>, data: &Array<V, (Ix, Ix)>, k: u32) -> Vec<V> {
    let mut distances = Vec::with_capacity(k as usize);
    for i in 0..data.dim().0 {
        let mut closest = distance_squared(&data.subview(0, i), &means.subview(0, 0));
        for j in 0..means.dim().0 {
            let distance = distance_squared(&data.subview(0, i), &means.subview(0, j));
            if distance < closest {
                closest = distance;
            }
        }
        distances.push(closest);
    }
    return distances;
}

#[cfg(test)]
mod tests {
    extern crate ndarray;
    // extern crate csv;
    // use super::kmeans;
    use ndarray::Array;
    
    #[test]
    fn test_distance() {
        use super::distance_squared;
        let a = Array::from_vec(vec![1.0f32, 1.0f32]);
        let b = Array::from_vec(vec![2.0f32, 2.0f32]);
        let c = Array::from_vec(vec![1200.0f32, 1200.0f32]);
        assert_eq!(distance_squared(&a, &b), 2.0f32);
        assert_eq!(distance_squared(&a, &c), 2875202.0f32);
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
